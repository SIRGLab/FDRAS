from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .model.cross_attn_fusion import ErrorMapUNet, CrossModalFixerUNet
from .model.no_ct.cross_attn_no_ct import NoCTCrossAttnFixerUNet


def _flag(cfg: dict, key: str) -> bool:
    val = cfg.get(key)
    return bool(cfg.get("use_ct_input", True)) if val is None else bool(val)


def load_model(ckpt: str, device: torch.device) -> tuple[ErrorMapUNet, torch.nn.Module, dict]:
    data = torch.load(ckpt, map_location=device, weights_only=False)
    if "err_net" not in data or "fixer" not in data:
        raise KeyError(
            f"{ckpt} has keys {sorted(data)}; expected an FDRAS checkpoint with 'err_net' and 'fixer'."
        )
    cfg = data.get("cfg", {})
    norm = str(cfg.get("norm", "INSTANCE"))
    err_net = ErrorMapUNet(
        in_channels=2 if _flag(cfg, "err_use_ct_input") else 1,
        base_channels=int(cfg.get("err_base_channels", 8)),
        norm=norm,
    ).to(device)
    fixer_cls = NoCTCrossAttnFixerUNet if cfg.get("fixer_type") == "cross_attn_no_ct" else CrossModalFixerUNet
    fixer = fixer_cls(
        in_channels=2 if _flag(cfg, "fixer_use_ct_input") else 1,
        err_channels=2,
        base_channels=int(cfg.get("fix_base_channels", 8)),
        norm=norm,
    ).to(device)
    err_net.load_state_dict(data["err_net"], strict=True)
    fixer.load_state_dict(data["fixer"], strict=True)
    err_net.eval()
    fixer.eval()
    return err_net, fixer, cfg


def pad_to_divisible(vol: np.ndarray, k: int = 8) -> tuple[np.ndarray, tuple[int, int, int]]:
    dz, dy, dx = vol.shape
    pd = (k - dz % k) % k
    ph = (k - dy % k) % k
    pw = (k - dx % k) % k
    vol_p = np.pad(vol, ((0, pd), (0, ph), (0, pw)), mode="constant", constant_values=0)
    return vol_p, (pd, ph, pw)


def scale_ct(ct: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    # Matches ScaleIntensityRanged(a_min, a_max, 0, 1, clip=True) used in training (data.py).
    return ((np.clip(ct, a_min, a_max) - a_min) / (a_max - a_min)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--ct", required=True, type=str)
    parser.add_argument("--pred", required=True, type=str)
    parser.add_argument("--out_prob", required=True, type=str)
    parser.add_argument("--out_mask", required=True, type=str)
    parser.add_argument("--out_pred", type=str, default=None)
    parser.add_argument("--out_err_dir", type=str, default=None, help="Save sigmoid tree/SDF error maps here")
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--out_gt", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    err_net, fixer, cfg = load_model(args.ckpt, device)

    ct_img = nib.load(args.ct)
    pred_img = nib.load(args.pred)
    ct = ct_img.get_fdata(dtype=np.float32)
    ct = scale_ct(ct, float(cfg.get("ct_window_min", -1000.0)), float(cfg.get("ct_window_max", 300.0)))
    pred = np.clip(pred_img.get_fdata(dtype=np.float32), 0.0, 1.0)

    ct_p, _ = pad_to_divisible(ct, k=8)
    pred_p, _ = pad_to_divisible(pred, k=8)

    ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).to(device)
    pred_t = torch.from_numpy(pred_p).unsqueeze(0).unsqueeze(0).to(device)
    x_err = torch.cat([ct_t, pred_t], dim=1) if _flag(cfg, "err_use_ct_input") else pred_t
    x = torch.cat([ct_t, pred_t], dim=1) if _flag(cfg, "fixer_use_ct_input") else pred_t

    with torch.no_grad():
        eps = 1e-5
        pred_prob = torch.clamp(pred_t, eps, 1 - eps)
        pred_logit = torch.log(pred_prob / (1 - pred_prob))
        err_pred = sliding_window_inference(
            inputs=x_err,
            roi_size=tuple(int(v) for v in args.roi_size),
            sw_batch_size=int(args.sw_batch_size),
            predictor=err_net,
            overlap=float(args.overlap),
        )
        combined = torch.cat([x, err_pred], dim=1)
        delta = sliding_window_inference(
            inputs=combined,
            roi_size=tuple(int(v) for v in args.roi_size),
            sw_batch_size=int(args.sw_batch_size),
            predictor=lambda z: fixer(z[:, : x.shape[1]], z[:, x.shape[1] :]),
            overlap=float(args.overlap),
        )
        refined = torch.sigmoid(pred_logit + delta)
        prob = refined.cpu().numpy()[0, 0]
        mask = (prob > float(args.threshold)).astype(np.float32)
        err = torch.sigmoid(err_pred).cpu().numpy()[0]

    dz, dy, dx = ct.shape
    prob = prob[:dz, :dy, :dx]
    mask = mask[:dz, :dy, :dx]
    err = err[:, :dz, :dy, :dx]

    out_prob = Path(args.out_prob)
    out_mask = Path(args.out_mask)
    out_prob.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prob, ct_img.affine, ct_img.header), str(out_prob))
    nib.save(nib.Nifti1Image(mask, ct_img.affine, ct_img.header), str(out_mask))

    if args.out_err_dir:
        err_dir = Path(args.out_err_dir)
        err_dir.mkdir(parents=True, exist_ok=True)
        for idx, name in enumerate(("tree_err", "sdf_err")):
            nib.save(nib.Nifti1Image(err[idx], ct_img.affine, ct_img.header), str(err_dir / f"{name}.nii.gz"))

    if args.out_pred:
        out_pred = Path(args.out_pred)
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred, pred_img.affine, pred_img.header), str(out_pred))

    if args.gt and args.out_gt:
        gt_img = nib.load(args.gt)
        gt = gt_img.get_fdata(dtype=np.float32)
        out_gt = Path(args.out_gt)
        out_gt.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(gt, gt_img.affine, gt_img.header), str(out_gt))


if __name__ == "__main__":
    main()
