from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference
import pandas as pd

from .data import build_dataloaders, discover_cases, split_train_val_test, to_monai_items
from .model.cross_attn_fusion import ErrorMapUNet, CrossModalFixerUNet
from .model.error_cross_attn import ErrorMapCrossAttnCT
from .model.channel_fusion import ChannelFusionFixerUNet
from .model.no_fusion import NoFusionFixerUNet
from .model.no_ct.cross_attn_no_ct import NoCTCrossAttnFixerUNet
from ..atr_patchgan_refine.losses import ccf_loss, cldice_loss, dice_loss, l1_consistency


@dataclass
class TrainConfig:
    # Data
    data_root: str = "runs/CT_Training_Data/ATM_Data"
    ct_subdir: str = "imagesTr"
    pred_subdir: str = "nnUNet_masksTr"
    gt_subdir: str = "labelsTr"
    val_count: int = 60
    test_count: int = 60
    seed: int = 42

    # Optional external validation sets (e.g., AeroPath + ARIC)
    val_data_roots: list[str] | None = None
    val_ct_subdir: str = "imagesTr"
    val_pred_subdir: str = "nnUNet_masksTr"
    val_gt_subdir: str = "labelsTr"
    val_pred_subdir_alt: str | None = None  # e.g., "TfeNet_masksTr"
    train_sample_csv: str | None = None
    train_sample_metric: str = "Dice"
    train_sample_n: int = 125
    train_sample_bins: int = 5
    train_sample_seed: int = 42
    val_sample_csv: str | None = None
    val_sample_csvs: list[str] | None = None
    val_sample_metric: str = "Dice"
    val_sample_n: int = 30
    val_sample_ns: list[int] | None = None
    val_sample_bins: int = 5
    val_sample_seed: int = 42
    val_sample_exclude_ids: list[str] | None = None
    train_extra_subdirs: list[str] | None = None
    val_extra_subdirs: list[str] | None = None
    err_tree_key: str = "nnUNet_masksTr_TreeErrTr"
    err_sdf_key: str = "nnUNet_masksTr_SDFErrTr"

    # Overfit/debug
    train_case_ids: list[str] | None = None
    overfit_use_train_as_val: bool = False

    # CT normalization
    ct_window_min: float = -1000.0
    ct_window_max: float = 300.0

    # Patch & loader
    patch_size: tuple[int, int, int] = (128, 96, 144)
    batch_size: int = 1
    num_workers: int = 2
    use_pos_neg_crop: bool = False
    pos_ratio: int = 1
    neg_ratio: int = 1
    num_samples: int = 1
    flatten_samples: bool = False
    expand_epoch_samples: bool = False
    repeat_per_case: int = 1
    use_persistent_cache: bool = False
    cache_dir: str | None = None
    val_repeat_per_case: int = 1
    val_use_patch_eval: bool = False
    train_full_volume: bool = False
    val_full_volume: bool = False
    val_use_sliding_window: bool = False
    sw_overlap: float = 0.5
    sw_batch_size: int = 1
    train_use_sliding_window: bool = False
    train_sw_overlap: float = 0.5
    train_sw_batch_size: int = 1
    train_use_grid_patch: bool = False
    train_grid_overlap: float = 0.0
    train_use_rand_grid_patch: bool = False
    train_rand_grid_overlap: float = 0.0
    train_rand_grid_num_patches: int | None = None
    use_ct_input: bool = True
    err_use_ct_input: bool | None = None  # if None, falls back to use_ct_input
    fixer_use_ct_input: bool | None = None  # if None, falls back to use_ct_input
    err_net_type: str = "unet"  # "unet" or "cross_attn_ct"
    save_val_case_id: str | None = None
    save_val_out_dir: str | None = None
    save_val_every: int = 1
    save_val_prob: bool = True
    save_val_mask: bool = True
    save_val_err_pred: bool = True
    val_compute_full_volume_metrics: bool = True

    # Model
    base_channels: int = 16
    err_base_channels: int = 8
    fix_base_channels: int = 8
    fixer_type: str = "cross_attn"  # "cross_attn", "channel_fusion", "no_fusion", "cross_attn_no_ct"
    norm: str = "INSTANCE"
    num_res_units: int = 1
    vit_patch_size: tuple[int, int, int] = (16, 16, 16)
    vit_hidden_size: int = 768
    vit_mlp_dim: int = 3072
    vit_num_layers: int = 12
    vit_num_heads: int = 12
    vit_dropout: float = 0.0

    # Loss weights
    w_l1: float = 0.1
    w_dice: float = 0.3
    w_bce: float = 0.0
    w_err_l1: float = 1.0
    w_err_bce: float = 1.0
    w_cldice: float = 0.3
    w_ccf: float = 0.6
    w_consistency: float = 0.0
    w_gan: float = 0.0
    cldice_iters: int = 3
    ccf_iters: int = 3

    # Optimization
    epochs: int = 150
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    weight_decay: float = 1e-5
    val_every: int = 1
    val_max_batches: int = 0

    # Runtime
    amp: bool = True
    device: str = "cuda"

    # Logging/checkpointing
    out_dir: str = "runs/repair_model/atr_refine_vit"
    save_every: int = 5
    use_tqdm: bool = True
    show_patch_progress: bool = True
    log_every_patches: int = 25
    resume_ckpt: str | None = None

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "atr-refine-vit"
    wandb_entity: str = "francis-xiatian-zhang-sirg"
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None


def load_cfg(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "patch_size" in raw and isinstance(raw["patch_size"], list):
        raw["patch_size"] = tuple(int(v) for v in raw["patch_size"])
    return TrainConfig(**raw)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("atr_refine_no_gan_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_cases_by_metric(
    cases: list,
    csv_path: str,
    metric: str,
    n: int,
    bins: int,
    seed: int,
) -> list:
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {csv_path}")
    df = df.dropna(subset=[metric])
    case_ids = {c.case_id for c in cases}
    df = df[df["case_id"].isin(case_ids)]
    if df.empty:
        return []
    df["bin"] = pd.qcut(df[metric], q=min(bins, df[metric].nunique()), duplicates="drop")
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in df.groupby("bin", observed=False):
        k = max(1, int(round(n / max(1, df["bin"].nunique()))))
        choices = group.sample(n=min(k, len(group)), random_state=seed)
        selected.extend(choices["case_id"].tolist())
    if len(selected) < n:
        remaining = df[~df["case_id"].isin(selected)]
        if not remaining.empty:
            extra = remaining.sample(n=min(n - len(selected), len(remaining)), random_state=seed)["case_id"].tolist()
            selected.extend(extra)
    selected = selected[:n]
    return [c for c in cases if c.case_id in set(selected)]

def _estimate_patches_per_item(dataset) -> int:
    try:
        sample = dataset[0]
    except Exception:
        return 1
    if isinstance(sample, list):
        return max(1, len(sample))
    return 1

def _estimate_grid_patches_for_case(ct_path: Path, patch_size: tuple[int, int, int], overlap: float) -> int:
    img = nib.load(str(ct_path))
    dz, dy, dx = img.shape
    strides = [max(1, int(ps * (1.0 - overlap))) for ps in patch_size]
    counts = []
    for dim, ps, st in zip((dz, dy, dx), patch_size, strides):
        if dim <= ps:
            counts.append(1)
        else:
            counts.append(1 + int(np.ceil((dim - ps) / st)))
    return int(counts[0] * counts[1] * counts[2])

def pad_to_divisible(vol: np.ndarray, k: int = 8) -> tuple[np.ndarray, tuple[int, int, int]]:
    dz, dy, dx = vol.shape
    pd = (k - dz % k) % k
    ph = (k - dy % k) % k
    pw = (k - dx % k) % k
    vol_p = np.pad(vol, ((0, pd), (0, ph), (0, pw)), mode="constant", constant_values=0)
    return vol_p, (pd, ph, pw)

def save_val_case_outputs(
    err_net: torch.nn.Module,
    fixer: torch.nn.Module,
    case,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    root_override: Path | None = None,
) -> float:
    if not cfg.save_val_out_dir:
        return 0.0
    out_dir = Path(cfg.save_val_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if root_override is not None:
        ct_path = root_override / cfg.val_ct_subdir / f"{case.case_id}.nii.gz"
        pred_path = root_override / cfg.val_pred_subdir / f"{case.case_id}.nii.gz"
        gt_path = root_override / cfg.val_gt_subdir / f"{case.case_id}.nii.gz"
    else:
        ct_path = case.ct_path
        pred_path = case.pred_path
        gt_path = case.gt_path

    ct_img = nib.load(str(ct_path))
    pred_img = nib.load(str(pred_path))
    ct = ct_img.get_fdata(dtype=np.float32)
    pred = np.clip(pred_img.get_fdata(dtype=np.float32), 0.0, 1.0)

    ct_p, pad = pad_to_divisible(ct, k=8)
    pred_p, _ = pad_to_divisible(pred, k=8)
    pd, ph, pw = pad

    ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).to(device)
    pred_t = torch.from_numpy(pred_p).unsqueeze(0).unsqueeze(0).to(device)
    err_use_ct = cfg.use_ct_input if cfg.err_use_ct_input is None else cfg.err_use_ct_input
    fix_use_ct = cfg.use_ct_input if cfg.fixer_use_ct_input is None else cfg.fixer_use_ct_input
    x_err = torch.cat([ct_t, pred_t], dim=1) if err_use_ct else pred_t
    x_main = torch.cat([ct_t, pred_t], dim=1) if fix_use_ct else pred_t

    # Error maps are not required for inference; they are only used for training supervision.

    with torch.no_grad():
        eps = 1e-5
        pred_prob = torch.clamp(pred_t, eps, 1 - eps)
        pred_logit = torch.log(pred_prob / (1 - pred_prob))
        if cfg.val_use_sliding_window:
            if cfg.err_net_type == "cross_attn_ct":
                if x_err.shape[1] != 2:
                    raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                err_pred = sliding_window_inference(
                    inputs=x_err,
                    roi_size=cfg.patch_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=lambda z: err_net(z[:, 1:2], z[:, 0:1]),
                    overlap=cfg.sw_overlap,
                )
            else:
                err_pred = sliding_window_inference(
                    inputs=x_err,
                    roi_size=cfg.patch_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=err_net,
                    overlap=cfg.sw_overlap,
                )
            combined = torch.cat([x_main, err_pred], dim=1)
            delta = sliding_window_inference(
                inputs=combined,
                roi_size=cfg.patch_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=lambda z: fixer(z[:, : x_main.shape[1]], z[:, x_main.shape[1] :]),
                overlap=cfg.sw_overlap,
            )
        else:
            if cfg.err_net_type == "cross_attn_ct":
                if x_err.shape[1] != 2:
                    raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                err_pred = err_net(x_err[:, 1:2], x_err[:, 0:1])
            else:
                err_pred = err_net(x_err)
            delta = fixer(x_main, err_pred)
        prob = torch.sigmoid(pred_logit + delta)

    prob_np = prob.cpu().numpy()[0, 0]
    if pd or ph or pw:
        dz, dy, dx = ct.shape
        prob_np = prob_np[:dz, :dy, :dx]

    if cfg.save_val_err_pred:
        err_np = err_pred.cpu().numpy()[0]
        if pd or ph or pw:
            dz, dy, dx = ct.shape
            err_np = err_np[:, :dz, :dy, :dx]
        out_tree = out_dir / f"epoch_{epoch:03d}_{case.case_id}_tree_err_pred.nii.gz"
        out_sdf = out_dir / f"epoch_{epoch:03d}_{case.case_id}_sdf_err_pred.nii.gz"
        nib.save(nib.Nifti1Image(err_np[0].astype(np.float32), ct_img.affine, ct_img.header), str(out_tree))
        nib.save(nib.Nifti1Image(err_np[1].astype(np.float32), ct_img.affine, ct_img.header), str(out_sdf))

    if cfg.save_val_prob:
        out_prob = out_dir / f"epoch_{epoch:03d}_{case.case_id}_prob.nii.gz"
        nib.save(nib.Nifti1Image(prob_np, ct_img.affine, ct_img.header), str(out_prob))
    if cfg.save_val_mask:
        mask = (prob_np > 0.5).astype(np.float32)
        out_mask = out_dir / f"epoch_{epoch:03d}_{case.case_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, ct_img.affine, ct_img.header), str(out_mask))
    else:
        mask = (prob_np > 0.5).astype(np.float32)

    # Optional: also save outputs using alt input mask (e.g., TfeNet)
    if root_override is not None and cfg.val_pred_subdir_alt:
        alt_pred_path = root_override / cfg.val_pred_subdir_alt / f"{case.case_id}.nii.gz"
        if alt_pred_path.exists():
            pred_alt = np.clip(nib.load(str(alt_pred_path)).get_fdata(dtype=np.float32), 0.0, 1.0)
            prob_alt = _infer_full_volume_prob(err_net, fixer, ct, pred_alt, cfg, device)
            if cfg.save_val_prob:
                out_prob_alt = out_dir / f"epoch_{epoch:03d}_{case.case_id}_tfenet_prob.nii.gz"
                nib.save(nib.Nifti1Image(prob_alt.astype(np.float32), ct_img.affine, ct_img.header), str(out_prob_alt))
            if cfg.save_val_mask:
                mask_alt = (prob_alt > 0.5).astype(np.float32)
                out_mask_alt = out_dir / f"epoch_{epoch:03d}_{case.case_id}_tfenet_mask.nii.gz"
                nib.save(nib.Nifti1Image(mask_alt, ct_img.affine, ct_img.header), str(out_mask_alt))

    gt = nib.load(str(gt_path)).get_fdata(dtype=np.float32) == 1
    dice = (2.0 * (mask > 0.5) * gt).sum() / ((mask > 0.5).sum() + gt.sum() + 1e-6)
    return float(dice)


def _infer_full_volume_prob(
    err_net: torch.nn.Module,
    fixer: torch.nn.Module,
    ct: np.ndarray,
    pred: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    ct_p, pad = pad_to_divisible(ct, k=8)
    pred_p, _ = pad_to_divisible(pred, k=8)
    pd, ph, pw = pad

    ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).to(device)
    pred_t = torch.from_numpy(pred_p).unsqueeze(0).unsqueeze(0).to(device)
    err_use_ct = cfg.use_ct_input if cfg.err_use_ct_input is None else cfg.err_use_ct_input
    fix_use_ct = cfg.use_ct_input if cfg.fixer_use_ct_input is None else cfg.fixer_use_ct_input
    x_err = torch.cat([ct_t, pred_t], dim=1) if err_use_ct else pred_t
    x_main = torch.cat([ct_t, pred_t], dim=1) if fix_use_ct else pred_t

    with torch.no_grad():
        eps = 1e-5
        pred_prob = torch.clamp(pred_t, eps, 1 - eps)
        pred_logit = torch.log(pred_prob / (1 - pred_prob))
        if cfg.val_use_sliding_window:
            if cfg.err_net_type == "cross_attn_ct":
                if x_err.shape[1] != 2:
                    raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                err_pred = sliding_window_inference(
                    inputs=x_err,
                    roi_size=cfg.patch_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=lambda z: err_net(z[:, 1:2], z[:, 0:1]),
                    overlap=cfg.sw_overlap,
                )
            else:
                err_pred = sliding_window_inference(
                    inputs=x_err,
                    roi_size=cfg.patch_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=err_net,
                    overlap=cfg.sw_overlap,
                )
            combined = torch.cat([x_main, err_pred], dim=1)
            delta = sliding_window_inference(
                inputs=combined,
                roi_size=cfg.patch_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=lambda z: fixer(z[:, : x_main.shape[1]], z[:, x_main.shape[1] :]),
                overlap=cfg.sw_overlap,
            )
        else:
            if cfg.err_net_type == "cross_attn_ct":
                if x_err.shape[1] != 2:
                    raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                err_pred = err_net(x_err[:, 1:2], x_err[:, 0:1])
            else:
                err_pred = err_net(x_err)
            delta = fixer(x_main, err_pred)
        prob = torch.sigmoid(pred_logit + delta).cpu().numpy()[0, 0]

    if pd or ph or pw:
        dz, dy, dx = ct.shape
        prob = prob[:dz, :dy, :dx]
    return prob


@torch.no_grad()
def validate(
    err_net: torch.nn.Module,
    fixer: torch.nn.Module,
    loader,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    err_net.eval()
    fixer.eval()
    dice_vals: list[float] = []
    if cfg.use_tqdm:
        total_val = max(1, len(loader.dataset))
        iter_loader = tqdm(loader, desc=f"val epoch {epoch}", leave=False, total=total_val)
    else:
        iter_loader = loader
    for idx, batch in enumerate(iter_loader, start=1):
        ct = batch["ct"].to(device)
        pred = batch["pred"].to(device)
        gt = (batch["gt"].to(device) == 1).float()

        err_use_ct = cfg.use_ct_input if cfg.err_use_ct_input is None else cfg.err_use_ct_input
        fix_use_ct = cfg.use_ct_input if cfg.fixer_use_ct_input is None else cfg.fixer_use_ct_input
        x_err = torch.cat([ct, pred], dim=1) if err_use_ct else pred
        x_main = torch.cat([ct, pred], dim=1) if fix_use_ct else pred
        eps = 1e-5
        pred_prob = torch.clamp(pred, eps, 1 - eps)
        pred_logit = torch.log(pred_prob / (1 - pred_prob))
        if cfg.val_use_sliding_window:
            if cfg.err_net_type == "cross_attn_ct":
                if x_err.shape[1] != 2:
                    raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                err_pred = sliding_window_inference(
                    inputs=x_err,
                    roi_size=cfg.patch_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=lambda z: err_net(z[:, 1:2], z[:, 0:1]),
                    overlap=cfg.sw_overlap,
                )
            else:
                err_pred = sliding_window_inference(
                    inputs=x_err,
                    roi_size=cfg.patch_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=err_net,
                    overlap=cfg.sw_overlap,
                )
            combined = torch.cat([x_main, err_pred], dim=1)
            delta = sliding_window_inference(
                inputs=combined,
                roi_size=cfg.patch_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=lambda z: fixer(z[:, : x_main.shape[1]], z[:, x_main.shape[1] :]),
                overlap=cfg.sw_overlap,
            )
        else:
            if cfg.err_net_type == "cross_attn_ct":
                if x_err.shape[1] != 2:
                    raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                err_pred = err_net(x_err[:, 1:2], x_err[:, 0:1])
            else:
                err_pred = err_net(x_err)
            delta = fixer(x_main, err_pred)
        prob = torch.sigmoid(pred_logit + delta)
        pred_bin = (prob > 0.5).float()
        dice_vals.append(float(1.0 - dice_loss(pred_bin, gt).item()))
        if cfg.val_max_batches and idx >= cfg.val_max_batches:
            break
    return {"val_dice": float(np.mean(dice_vals)) if dice_vals else 0.0}


def run_training(cfg: TrainConfig) -> None:
    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)

    data_root = Path(cfg.data_root)
    wandb_run = None
    if cfg.use_wandb:
        try:
            if "WANDB_API_KEY" not in os.environ:
                key_path = Path("runs") / "wandb_key"
                if key_path.exists():
                    os.environ["WANDB_API_KEY"] = key_path.read_text(encoding="utf-8").strip()
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                tags=cfg.wandb_tags,
                config=asdict(cfg),
            )
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)
    cases = discover_cases(data_root, cfg.ct_subdir, cfg.pred_subdir, cfg.gt_subdir)
    train_cases, val_cases, test_cases = split_train_val_test(
        cases, val_count=cfg.val_count, test_count=cfg.test_count, seed=cfg.seed
    )
    logger.info("Data root: %s", data_root)
    logger.info(
        "Cases: total=%d train=%d val=%d test=%d",
        len(cases),
        len(train_cases),
        len(val_cases),
        len(test_cases),
    )

    if cfg.train_case_ids:
        train_cases = [c for c in train_cases if c.case_id in set(cfg.train_case_ids)]
        logger.info("Overfit train cases: %s", [c.case_id for c in train_cases])

    if cfg.train_sample_csv:
        train_cases = sample_cases_by_metric(
            train_cases,
            cfg.train_sample_csv,
            cfg.train_sample_metric,
            cfg.train_sample_n,
            cfg.train_sample_bins,
            cfg.train_sample_seed,
        )
    elif cfg.train_sample_n and cfg.train_sample_n > 0:
        rng = random.Random(cfg.train_sample_seed)
        if cfg.train_sample_n < len(train_cases):
            train_cases = rng.sample(train_cases, cfg.train_sample_n)
        logger.info("Sampled train cases: %d", len(train_cases))
    train_items = to_monai_items(train_cases, extra_subdirs=cfg.train_extra_subdirs)
    if cfg.overfit_use_train_as_val:
        val_items = to_monai_items(train_cases, extra_subdirs=cfg.val_extra_subdirs)
    elif cfg.val_data_roots:
        val_items = []
        val_ids = []
        for i, root in enumerate(cfg.val_data_roots):
            v_cases = discover_cases(
                Path(root),
                cfg.val_ct_subdir,
                cfg.val_pred_subdir,
                cfg.val_gt_subdir,
            )
            csv_path = None
            if cfg.val_sample_csvs and i < len(cfg.val_sample_csvs):
                csv_path = cfg.val_sample_csvs[i]
            elif cfg.val_sample_csv:
                csv_path = cfg.val_sample_csv
            sample_n = cfg.val_sample_n
            if cfg.val_sample_ns and i < len(cfg.val_sample_ns):
                sample_n = cfg.val_sample_ns[i]
            if csv_path:
                v_cases = sample_cases_by_metric(
                    v_cases,
                    csv_path,
                    cfg.val_sample_metric,
                    sample_n,
                    cfg.val_sample_bins,
                    cfg.val_sample_seed,
                )
            if cfg.val_sample_exclude_ids:
                v_cases = [c for c in v_cases if c.case_id not in set(cfg.val_sample_exclude_ids)]
            val_ids.extend([c.case_id for c in v_cases])
            val_items.extend(to_monai_items(v_cases, extra_subdirs=cfg.val_extra_subdirs))
        logger.info("Val case IDs: %s", val_ids)
        if cfg.save_val_case_id is None and val_ids:
            cfg.save_val_case_id = val_ids[0]
            logger.info("Auto-selected save_val_case_id: %s", cfg.save_val_case_id)
    else:
        val_items = to_monai_items(val_cases, extra_subdirs=cfg.val_extra_subdirs)
    train_loader, val_loader = build_dataloaders(
        train_items=train_items,
        val_items=val_items,
        patch_size=cfg.patch_size,
        ct_window_min=cfg.ct_window_min,
        ct_window_max=cfg.ct_window_max,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_pos_neg_crop=cfg.use_pos_neg_crop,
        pos_ratio=cfg.pos_ratio,
        neg_ratio=cfg.neg_ratio,
        num_samples=cfg.num_samples,
        flatten_samples=cfg.flatten_samples,
        expand_epoch_samples=cfg.expand_epoch_samples,
        repeat_per_case=cfg.repeat_per_case,
        use_persistent_cache=cfg.use_persistent_cache,
        cache_dir=cfg.cache_dir,
        val_repeat_per_case=cfg.val_repeat_per_case,
        val_use_patch_eval=cfg.val_use_patch_eval,
        train_full_volume=cfg.train_full_volume,
        val_full_volume=cfg.val_full_volume,
        train_use_grid_patch=cfg.train_use_grid_patch,
        train_grid_overlap=cfg.train_grid_overlap,
        train_use_rand_grid_patch=cfg.train_use_rand_grid_patch,
        train_rand_grid_overlap=cfg.train_rand_grid_overlap,
        train_rand_grid_num_patches=cfg.train_rand_grid_num_patches,
        extra_keys_train=cfg.train_extra_subdirs,
        extra_keys_val=cfg.val_extra_subdirs,
        seed=cfg.seed,
    )

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    err_in = 2 if (cfg.err_use_ct_input if cfg.err_use_ct_input is not None else cfg.use_ct_input) else 1
    fix_in = 2 if (cfg.fixer_use_ct_input if cfg.fixer_use_ct_input is not None else cfg.use_ct_input) else 1
    if cfg.err_net_type == "cross_attn_ct":
        err_net = ErrorMapCrossAttnCT(
            mask_channels=1,
            ct_channels=1,
            base_channels=cfg.err_base_channels,
            norm=cfg.norm,
        ).to(device)
    else:
        err_net = ErrorMapUNet(
            in_channels=err_in,
            base_channels=cfg.err_base_channels,
            norm=cfg.norm,
        ).to(device)
    if cfg.fixer_type == "channel_fusion":
        fixer = ChannelFusionFixerUNet(
            in_channels=fix_in,
            err_channels=2,
            base_channels=cfg.fix_base_channels,
            norm=cfg.norm,
        ).to(device)
    elif cfg.fixer_type == "cross_attn_no_ct":
        fixer = NoCTCrossAttnFixerUNet(
            in_channels=fix_in,
            err_channels=2,
            base_channels=cfg.fix_base_channels,
            norm=cfg.norm,
        ).to(device)
    elif cfg.fixer_type == "no_fusion":
        fixer = NoFusionFixerUNet(
            in_channels=fix_in,
            base_channels=cfg.fix_base_channels,
            norm=cfg.norm,
        ).to(device)
    else:
        fixer = CrossModalFixerUNet(
            in_channels=fix_in,
            err_channels=2,
            base_channels=cfg.fix_base_channels,
            norm=cfg.norm,
        ).to(device)

    opt_g = torch.optim.AdamW(
        list(err_net.parameters()) + list(fixer.parameters()),
        lr=cfg.lr_g,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = -1.0
    start_epoch = 1
    if cfg.resume_ckpt:
        ckpt_path = Path(cfg.resume_ckpt)
        if ckpt_path.exists():
            data = torch.load(str(ckpt_path), map_location=device)
            if "err_net" in data:
                err_net.load_state_dict(data["err_net"], strict=True)
            if "fixer" in data:
                fixer.load_state_dict(data["fixer"], strict=True)
            if "opt_g" in data:
                opt_g.load_state_dict(data["opt_g"])
            # try to parse epoch from filename if present
            try:
                name = ckpt_path.stem  # e.g., epoch_5
                if name.startswith("epoch_"):
                    start_epoch = int(name.split("_")[1]) + 1
            except Exception:
                start_epoch = 1
            logger.info("Resumed from %s (start_epoch=%d)", ckpt_path, start_epoch)
        else:
            logger.warning("resume_ckpt not found: %s", ckpt_path)
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            epoch_t0 = time.perf_counter()
            err_net.train()
            fixer.train()
            planned_patches = None
            if cfg.expand_epoch_samples:
                planned_patches = max(1, len(train_cases) * cfg.repeat_per_case)
            elif cfg.use_pos_neg_crop and cfg.num_samples > 1:
                planned_patches = max(1, len(train_cases) * cfg.num_samples)
            elif cfg.train_use_rand_grid_patch and cfg.train_rand_grid_num_patches:
                planned_patches = max(1, cfg.train_rand_grid_num_patches * len(train_cases))
            elif cfg.train_use_grid_patch:
                try:
                    total_per_case = _estimate_grid_patches_for_case(
                        train_cases[0].ct_path, cfg.patch_size, cfg.train_grid_overlap
                    )
                    planned_patches = max(1, total_per_case * len(train_cases))
                except Exception:
                    planned_patches = None
            if planned_patches is None:
                planned_patches = max(1, len(train_loader))
            if cfg.use_tqdm and cfg.show_patch_progress:
                pbar = tqdm(total=planned_patches, desc=f"train epoch {epoch}", unit="patch")
                iter_loader = train_loader
            else:
                pbar = None
                iter_loader = tqdm(train_loader, desc=f"train epoch {epoch}") if cfg.use_tqdm else train_loader

            loss_sum = 0.0
            batch_count = 0
            next_patch_log = max(1, int(cfg.log_every_patches))
            for batch in iter_loader:
                ct = batch["ct"].to(device)
                pred = torch.clamp(batch["pred"].to(device), 0.0, 1.0)
                gt = (batch["gt"].to(device) == 1).float()
                tree_gt = torch.clamp(batch[cfg.err_tree_key].to(device), 0.0, 1.0)
                sdf_gt = torch.clamp(batch[cfg.err_sdf_key].to(device), 0.0, 1.0)

                # RandGridPatchd may return an extra patches dimension: [B, N, C, D, H, W]
                if ct.dim() == 6:
                    b, n, c, d, h, w = ct.shape
                    ct = ct.view(b * n, c, d, h, w)
                    pred = pred.view(b * n, c, d, h, w)
                    gt = gt.view(b * n, c, d, h, w)
                    tree_gt = tree_gt.view(b * n, c, d, h, w)
                    sdf_gt = sdf_gt.view(b * n, c, d, h, w)

                bs = int(ct.shape[0])
                batch_count += bs

                err_use_ct = cfg.use_ct_input if cfg.err_use_ct_input is None else cfg.err_use_ct_input
                fix_use_ct = cfg.use_ct_input if cfg.fixer_use_ct_input is None else cfg.fixer_use_ct_input
                x_err = torch.cat([ct, pred], dim=1) if err_use_ct else pred
                x_main = torch.cat([ct, pred], dim=1) if fix_use_ct else pred
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    # Error-map prediction
                    if cfg.err_net_type == "cross_attn_ct":
                        if x_err.shape[1] != 2:
                            raise ValueError("err_net_type=cross_attn_ct requires CT+mask input.")
                        err_pred = err_net(x_err[:, 1:2], x_err[:, 0:1])
                    else:
                        err_pred = err_net(x_err)
                    err_gt = torch.cat([tree_gt, sdf_gt], dim=1)
                    loss_err_l1 = F.l1_loss(err_pred, err_gt)
                    loss_err_bce = F.binary_cross_entropy_with_logits(err_pred, err_gt)

                    # Fixer (logit residual)
                    eps = 1e-5
                    pred_prob = torch.clamp(pred, eps, 1 - eps)
                    pred_logit = torch.log(pred_prob / (1 - pred_prob))
                    delta = fixer(x_main, err_pred)
                    refined_logit = pred_logit + delta
                    refined_prob = torch.sigmoid(refined_logit)

                    loss_l1 = F.l1_loss(refined_prob, gt)
                    loss_dice = dice_loss(refined_prob, gt)
                    loss_bce = F.binary_cross_entropy_with_logits(refined_logit, gt)
                    loss_cl = cldice_loss(refined_prob, gt, iters=cfg.cldice_iters) if cfg.w_cldice > 0 else 0.0
                    loss_ccf = ccf_loss(refined_prob, gt, iters=cfg.ccf_iters) if cfg.w_ccf > 0 else 0.0
                    loss_cons = l1_consistency(refined_prob, pred) if cfg.w_consistency > 0 else 0.0
                    loss = (
                        cfg.w_err_l1 * loss_err_l1
                        + cfg.w_err_bce * loss_err_bce
                        + cfg.w_l1 * loss_l1
                        + cfg.w_dice * loss_dice
                        + cfg.w_bce * loss_bce
                        + (cfg.w_cldice * loss_cl if cfg.w_cldice > 0 else 0.0)
                        + (cfg.w_ccf * loss_ccf if cfg.w_ccf > 0 else 0.0)
                        + (cfg.w_consistency * loss_cons if cfg.w_consistency > 0 else 0.0)
                    )

                scaler.scale(loss).backward()
                scaler.step(opt_g)
                scaler.update()
                opt_g.zero_grad(set_to_none=True)

                loss_sum += float(loss.item()) * bs
                if pbar is not None:
                    pbar.update(bs)
                    pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
                elif cfg.use_tqdm:
                    iter_loader.set_postfix(loss=f"{float(loss.item()):.4f}")

                if cfg.log_every_patches > 0:
                    while batch_count >= next_patch_log:
                        pct = 100.0 * float(next_patch_log) / float(max(1, planned_patches))
                        logger.info(
                            "epoch=%d patch=%d/%d (%.2f%%) loss=%.6f",
                            epoch,
                            next_patch_log,
                            planned_patches,
                            pct,
                            float(loss.item()),
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "epoch": epoch,
                                    "train_patch": int(next_patch_log),
                                    "train_patch_total": int(planned_patches),
                                    "train_patch_pct": float(pct),
                                    "train_patch_loss": float(loss.item()),
                                }
                            )
                        next_patch_log += max(1, int(cfg.log_every_patches))
            if pbar is not None:
                pbar.close()

            train_loss = loss_sum / max(1, batch_count)
            train_time_sec = time.perf_counter() - epoch_t0
            if batch_count == 0:
                logger.warning("No training batches were processed this epoch.")

            if cfg.val_every > 0 and (epoch % cfg.val_every == 0 or epoch == cfg.epochs):
                val_t0 = time.perf_counter()
                metrics = validate(err_net, fixer, val_loader, cfg, device, epoch)
                val_time_sec = time.perf_counter() - val_t0
                # Full-volume per-case dice for nnUNet input and optional TfeNet input
                if cfg.val_data_roots and cfg.val_compute_full_volume_metrics:
                    csv_path = Path(cfg.out_dir) / "val_full_volume_dice.csv"
                    rows = []
                    for root in cfg.val_data_roots:
                        root_p = Path(root)
                        ct_dir = root_p / cfg.val_ct_subdir
                        pred_dir = root_p / cfg.val_pred_subdir
                        gt_dir = root_p / cfg.val_gt_subdir
                        pred_alt_dir = root_p / cfg.val_pred_subdir_alt if cfg.val_pred_subdir_alt else None
                        for pred_path in sorted(pred_dir.glob("*.nii*")):
                            case_id = pred_path.name.replace(".nii.gz", "").replace(".nii", "")
                            if "val_ids" in locals() and val_ids and case_id not in val_ids:
                                continue
                            ct_path = ct_dir / pred_path.name
                            gt_path = gt_dir / pred_path.name
                            if not ct_path.exists() or not gt_path.exists():
                                continue
                            ct_img = nib.load(str(ct_path))
                            ct = ct_img.get_fdata(dtype=np.float32)
                            gt = nib.load(str(gt_path)).get_fdata(dtype=np.float32) == 1
                            pred = np.clip(nib.load(str(pred_path)).get_fdata(dtype=np.float32), 0.0, 1.0)
                            prob = _infer_full_volume_prob(err_net, fixer, ct, pred, cfg, device)
                            mask = prob > 0.5
                            dice_refined = (2.0 * (mask & gt).sum()) / ((mask).sum() + gt.sum() + 1e-6)
                            dice_refined_alt = np.nan
                            if pred_alt_dir:
                                alt_path = pred_alt_dir / pred_path.name
                                if alt_path.exists():
                                    pred_alt = np.clip(nib.load(str(alt_path)).get_fdata(dtype=np.float32), 0.0, 1.0)
                                    prob_alt = _infer_full_volume_prob(err_net, fixer, ct, pred_alt, cfg, device)
                                    mask_alt = prob_alt > 0.5
                                    dice_refined_alt = (2.0 * (mask_alt & gt).sum()) / ((mask_alt).sum() + gt.sum() + 1e-6)
                            rows.append(
                                {
                                    "epoch": epoch,
                                    "case_id": case_id,
                                    "dice_refined_nnunet": float(dice_refined),
                                    "dice_refined_tfenet": float(dice_refined_alt)
                                    if np.isfinite(dice_refined_alt)
                                    else np.nan,
                                }
                            )
                    if rows:
                        df = pd.DataFrame(rows)
                        header = not csv_path.exists()
                        df.to_csv(csv_path, mode="a", header=header, index=False)
                extra_metrics = {}
                if (
                    cfg.val_compute_full_volume_metrics
                    and cfg.save_val_case_id
                    and cfg.save_val_every > 0
                    and epoch % cfg.save_val_every == 0
                ):
                    # Prefer external val roots if the case is in val_ids
                    handled = False
                    if cfg.val_data_roots and "val_ids" in locals() and val_ids and cfg.save_val_case_id in val_ids:
                        for root in cfg.val_data_roots:
                            root_p = Path(root)
                            ct_path = root_p / cfg.val_ct_subdir / f"{cfg.save_val_case_id}.nii.gz"
                            if ct_path.exists():
                                dummy = type("C", (), {"case_id": cfg.save_val_case_id})()
                                fv_dice = save_val_case_outputs(
                                    err_net, fixer, dummy, cfg, device, epoch, root_override=root_p
                                )
                                extra_metrics["val_full_volume_dice_nnunet"] = fv_dice
                                if cfg.val_pred_subdir_alt:
                                    alt_pred_dir = root_p / cfg.val_pred_subdir_alt
                                    alt_path = alt_pred_dir / f"{cfg.save_val_case_id}.nii.gz"
                                    if alt_path.exists():
                                        ct_img = nib.load(str(ct_path))
                                        ct = ct_img.get_fdata(dtype=np.float32)
                                        pred_alt = np.clip(
                                            nib.load(str(alt_path)).get_fdata(dtype=np.float32), 0.0, 1.0
                                        )
                                        prob_alt = _infer_full_volume_prob(err_net, fixer, ct, pred_alt, cfg, device)
                                        mask_alt = prob_alt > 0.5
                                        gt = (
                                            nib.load(
                                                str(root_p / cfg.val_gt_subdir / f"{cfg.save_val_case_id}.nii.gz")
                                            )
                                            .get_fdata(dtype=np.float32)
                                            == 1
                                        )
                                        dice_alt = (2.0 * (mask_alt & gt).sum()) / (
                                            (mask_alt).sum() + gt.sum() + 1e-6
                                        )
                                        extra_metrics["val_full_volume_dice_tfenet"] = float(dice_alt)
                                handled = True
                                break
                    if not handled:
                        case_map = {c.case_id: c for c in cases}
                        case = case_map.get(cfg.save_val_case_id)
                        if case:
                            fv_dice = save_val_case_outputs(err_net, fixer, case, cfg, device, epoch)
                            extra_metrics["val_full_volume_dice_nnunet"] = fv_dice
                        elif cfg.val_data_roots:
                            for root in cfg.val_data_roots:
                                root_p = Path(root)
                                ct_path = root_p / cfg.val_ct_subdir / f"{cfg.save_val_case_id}.nii.gz"
                                if ct_path.exists():
                                    dummy = type("C", (), {"case_id": cfg.save_val_case_id})()
                                    fv_dice = save_val_case_outputs(
                                        err_net, fixer, dummy, cfg, device, epoch, root_override=root_p
                                    )
                                    extra_metrics["val_full_volume_dice_nnunet"] = fv_dice
                                    break
                logger.info(
                    "epoch=%d train_loss=%.5f val_dice=%.5f train_time_sec=%.2f val_time_sec=%.2f epoch_time_sec=%.2f",
                    epoch,
                    train_loss,
                    metrics["val_dice"],
                    train_time_sec,
                    val_time_sec,
                    train_time_sec + val_time_sec,
                )
                if metrics["val_dice"] > best_val:
                    best_val = metrics["val_dice"]
                    torch.save(
                        {
                            "err_net": err_net.state_dict(),
                            "fixer": fixer.state_dict(),
                            "opt_g": opt_g.state_dict(),
                            "cfg": asdict(cfg),
                        },
                        ckpt_dir / "best.pt",
                    )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss": train_loss,
                            "val_dice": metrics["val_dice"],
                            "train_time_sec": train_time_sec,
                            "val_time_sec": val_time_sec,
                            "epoch_time_sec": train_time_sec + val_time_sec,
                            "epoch": epoch,
                            **extra_metrics,
                        }
                    )
            else:
                logger.info("epoch=%d train_loss=%.5f train_time_sec=%.2f", epoch, train_loss, train_time_sec)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss": train_loss,
                            "train_time_sec": train_time_sec,
                            "epoch_time_sec": train_time_sec,
                            "epoch": epoch,
                        }
                    )

            if cfg.save_every > 0 and epoch % cfg.save_every == 0:
                torch.save(
                    {
                        "err_net": err_net.state_dict(),
                        "fixer": fixer.state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "cfg": asdict(cfg),
                    },
                    ckpt_dir / f"epoch_{epoch}.pt",
                )
    except Exception:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise

    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")
    (out_dir / "test_split.json").write_text(
        json.dumps({"test_cases": [c.case_id for c in test_cases]}, indent=2), encoding="utf-8"
    )
    logger.info("Training done. Best val_dice=%.5f", best_val)
    if wandb_run is not None:
        wandb_run.finish(exit_code=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
