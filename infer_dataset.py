from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference

from .model.cross_attn_fusion import ErrorMapUNet, CrossModalFixerUNet


def load_model(ckpt: str, device: torch.device) -> tuple[ErrorMapUNet, CrossModalFixerUNet, dict]:
    data = torch.load(ckpt, map_location=device)
    cfg = data.get("cfg", {})
    err_net = ErrorMapUNet(
        in_channels=2 if bool(cfg.get("use_ct_input", True)) else 1,
        base_channels=int(cfg.get("err_base_channels", 8)),
        norm=str(cfg.get("norm", "INSTANCE")),
    ).to(device)
    fixer = CrossModalFixerUNet(
        in_channels=2 if bool(cfg.get("use_ct_input", True)) else 1,
        err_channels=2,
        base_channels=int(cfg.get("fix_base_channels", 8)),
        norm=str(cfg.get("norm", "INSTANCE")),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--ct_subdir", default="imagesTr", type=str)
    parser.add_argument("--pred_subdir", default="nnUNet_masksTr", type=str)
    parser.add_argument("--gt_subdir", default=None, type=str)
    parser.add_argument("--out_root", required=True, type=str)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    err_net, fixer, cfg = load_model(args.ckpt, device)

    data_root = Path(args.data_root)
    ct_dir = data_root / args.ct_subdir
    pred_dir = data_root / args.pred_subdir
    out_root = Path(args.out_root)
    out_prob = out_root / "prob"
    out_mask = out_root / "mask"
    out_pred = out_root / "pred"
    out_gt = out_root / "gt"
    out_prob.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)
    if args.gt_subdir:
        out_gt.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_dir.glob("*.nii*"))
    for pred_path in tqdm(pred_paths, desc="infer"):
        ct_path = ct_dir / pred_path.name
        if not ct_path.exists():
            continue
        ct_img = nib.load(str(ct_path))
        pred_img = nib.load(str(pred_path))
        ct = ct_img.get_fdata(dtype=np.float32)
        pred = np.clip(pred_img.get_fdata(dtype=np.float32), 0.0, 1.0)

        ct_p, pad = pad_to_divisible(ct, k=8)
        pred_p, _ = pad_to_divisible(pred, k=8)
        pd, ph, pw = pad

        ct_t = torch.from_numpy(ct_p).unsqueeze(0).unsqueeze(0).to(device)
        pred_t = torch.from_numpy(pred_p).unsqueeze(0).unsqueeze(0).to(device)
        use_ct = bool(cfg.get("use_ct_input", True))
        x = torch.cat([ct_t, pred_t], dim=1) if use_ct else pred_t

        with torch.no_grad():
            eps = 1e-5
            pred_prob = torch.clamp(pred_t, eps, 1 - eps)
            pred_logit = torch.log(pred_prob / (1 - pred_prob))
            err_pred = sliding_window_inference(
                inputs=x,
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

        if pd or ph or pw:
            dz, dy, dx = ct.shape
            prob = prob[:dz, :dy, :dx]
            mask = mask[:dz, :dy, :dx]

        nib.save(nib.Nifti1Image(prob, ct_img.affine, ct_img.header), out_prob / pred_path.name)
        nib.save(nib.Nifti1Image(mask, ct_img.affine, ct_img.header), out_mask / pred_path.name)
        nib.save(nib.Nifti1Image(pred, pred_img.affine, pred_img.header), out_pred / pred_path.name)

        if args.gt_subdir:
            gt_path = data_root / args.gt_subdir / pred_path.name
            if gt_path.exists():
                gt_img = nib.load(str(gt_path))
                gt = gt_img.get_fdata(dtype=np.float32)
                nib.save(nib.Nifti1Image(gt, gt_img.affine, gt_img.header), out_gt / pred_path.name)


if __name__ == "__main__":
    main()
