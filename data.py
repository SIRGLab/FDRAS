from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import functools
import random

import torch
import numpy as np
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    GridPatchd,
    RandGridPatchd,
    ScaleIntensityRanged,
    SpatialPadd,
)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    ct_path: Path
    pred_path: Path
    gt_path: Path


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base, repeats: int) -> None:
        super().__init__()
        self._base = base
        self._repeats = max(1, int(repeats))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._base) * self._repeats

    def __getitem__(self, index):  # type: ignore[override]
        return self._base[index % len(self._base)]


def _seed_worker(worker_id: int, base_seed: int) -> None:
    # Deterministic worker seeding for reproducible random crops.
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _flatten_collate(batch):
    flat = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return torch.utils.data.dataloader.default_collate(flat)


def discover_cases(
    data_root: Path,
    ct_subdir: str,
    pred_subdir: str,
    gt_subdir: str,
) -> list[CaseRecord]:
    ct_dir = data_root / ct_subdir
    pred_dir = data_root / pred_subdir
    gt_dir = data_root / gt_subdir
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT dir not found: {ct_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    cases: list[CaseRecord] = []
    for ct_path in sorted(ct_dir.glob("*.nii*")):
        pred_path = pred_dir / ct_path.name
        gt_path = gt_dir / ct_path.name
        if not pred_path.exists() or not gt_path.exists():
            continue
        case_id = ct_path.name.replace(".nii.gz", "").replace(".nii", "")
        cases.append(CaseRecord(case_id=case_id, ct_path=ct_path, pred_path=pred_path, gt_path=gt_path))

    if not cases:
        raise RuntimeError(
            "No matched cases found between CT/pred/GT folders. "
            f"Checked: {ct_dir}, {pred_dir}, {gt_dir}"
        )
    return cases


def split_train_val_test(
    cases: Sequence[CaseRecord],
    val_count: int,
    test_count: int,
    seed: int,
) -> tuple[list[CaseRecord], list[CaseRecord], list[CaseRecord]]:
    if len(cases) < (val_count + test_count + 1):
        raise RuntimeError("Not enough cases for requested split.")
    idx = list(range(len(cases)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_idx = set(idx[:val_count])
    test_idx = set(idx[val_count : val_count + test_count])

    train_cases, val_cases, test_cases = [], [], []
    for i, c in enumerate(cases):
        if i in val_idx:
            val_cases.append(c)
        elif i in test_idx:
            test_cases.append(c)
        else:
            train_cases.append(c)
    return train_cases, val_cases, test_cases


def to_monai_items(cases: Sequence[CaseRecord], extra_subdirs: Sequence[str] | None = None) -> list[dict]:
    items: list[dict] = []
    for c in cases:
        item = {
            "case_id": c.case_id,
            "ct": str(c.ct_path),
            "pred": str(c.pred_path),
            "gt": str(c.gt_path),
        }
        if extra_subdirs:
            for sub in extra_subdirs:
                p = c.ct_path.parent.parent / sub / c.ct_path.name
                if p.exists():
                    item[sub] = str(p)
        items.append(item)
    return items


def build_transforms(
    patch_size: tuple[int, int, int],
    ct_window_min: float,
    ct_window_max: float,
    train: bool,
    use_pos_neg_crop: bool = False,
    pos_ratio: int = 1,
    neg_ratio: int = 1,
    num_samples: int = 1,
    force_no_crop: bool = False,
    use_grid_patch: bool = False,
    grid_overlap: float = 0.0,
    use_rand_grid_patch: bool = False,
    rand_grid_overlap: float = 0.0,
    rand_grid_num_patches: int | None = None,
    extra_keys: Sequence[str] | None = None,
):
    keys = ["ct", "pred", "gt"]
    if extra_keys:
        keys = keys + list(extra_keys)
    xforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(
            keys=["ct"],
            a_min=ct_window_min,
            a_max=ct_window_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        DivisiblePadd(keys=keys, k=8),
        SpatialPadd(keys=keys, spatial_size=patch_size),
    ]
    if train and not force_no_crop:
        if use_rand_grid_patch:
            xforms.append(
                RandGridPatchd(
                    keys=keys,
                    patch_size=patch_size,
                    overlap=rand_grid_overlap,
                    num_patches=rand_grid_num_patches,
                    pad_mode="constant",
                    constant_values=0,
                )
            )
        elif use_grid_patch:
            xforms.append(
                GridPatchd(
                    keys=keys,
                    patch_size=patch_size,
                    overlap=grid_overlap,
                    pad_mode="constant",
                    constant_values=0,
                )
            )
        elif use_pos_neg_crop:
            xforms.append(
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="gt",
                    spatial_size=patch_size,
                    pos=pos_ratio,
                    neg=neg_ratio,
                    num_samples=num_samples,
                    image_key="ct",
                )
            )
        else:
            xforms.append(
                RandSpatialCropd(
                    keys=keys,
                    roi_size=patch_size,
                    random_center=True,
                    random_size=False,
                )
            )
    xforms.append(EnsureTyped(keys=keys, dtype=torch.float32))
    return Compose(xforms)


def build_dataloaders(
    train_items: Sequence[dict],
    val_items: Sequence[dict],
    patch_size: tuple[int, int, int],
    ct_window_min: float,
    ct_window_max: float,
    batch_size: int,
    num_workers: int,
    use_pos_neg_crop: bool = False,
    pos_ratio: int = 1,
    neg_ratio: int = 1,
    num_samples: int = 1,
    flatten_samples: bool = False,
    expand_epoch_samples: bool = False,
    repeat_per_case: int = 1,
    use_persistent_cache: bool = False,
    cache_dir: str | None = None,
    val_repeat_per_case: int = 1,
    val_use_patch_eval: bool = False,
    train_full_volume: bool = False,
    val_full_volume: bool = False,
    train_use_grid_patch: bool = False,
    train_grid_overlap: float = 0.0,
    train_use_rand_grid_patch: bool = False,
    train_rand_grid_overlap: float = 0.0,
    train_rand_grid_num_patches: int | None = None,
    extra_keys_train: Sequence[str] | None = None,
    extra_keys_val: Sequence[str] | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    train_tf = build_transforms(
        patch_size,
        ct_window_min,
        ct_window_max,
        train=True,
        use_pos_neg_crop=use_pos_neg_crop,
        pos_ratio=pos_ratio,
        neg_ratio=neg_ratio,
        num_samples=1 if expand_epoch_samples else num_samples,
        force_no_crop=train_full_volume,
        use_grid_patch=train_use_grid_patch,
        grid_overlap=train_grid_overlap,
        use_rand_grid_patch=train_use_rand_grid_patch,
        rand_grid_overlap=train_rand_grid_overlap,
        rand_grid_num_patches=train_rand_grid_num_patches,
        extra_keys=extra_keys_train,
    )
    val_tf = build_transforms(
        patch_size,
        ct_window_min,
        ct_window_max,
        train=bool(val_use_patch_eval),
        use_pos_neg_crop=False,
        force_no_crop=(val_full_volume and not val_use_patch_eval),
        extra_keys=extra_keys_val,
    )

    def _make_dataset(items, transform):
        if use_persistent_cache:
            if not cache_dir:
                raise ValueError("cache_dir must be set when use_persistent_cache is true")
            return PersistentDataset(data=list(items), transform=transform, cache_dir=str(cache_dir))
        return Dataset(data=list(items), transform=transform)

    base_train_ds = _make_dataset(train_items, train_tf)
    train_ds = (
        RepeatDataset(base_train_ds, repeats=repeat_per_case)
        if expand_epoch_samples
        else base_train_ds
    )
    base_val_ds = _make_dataset(val_items, val_tf) if val_items else Dataset(data=list(val_items), transform=val_tf)
    val_ds = (
        RepeatDataset(base_val_ds, repeats=val_repeat_per_case)
        if val_repeat_per_case and val_repeat_per_case > 1
        else base_val_ds
    )

    worker_init = functools.partial(_seed_worker, base_seed=seed)

    if flatten_samples:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_flatten_collate,
            worker_init_fn=worker_init,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=False,
        worker_init_fn=worker_init,
    )
    return train_loader, val_loader
