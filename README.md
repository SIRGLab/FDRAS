# FDRAS: Failure Diagnosis and Repair for Airway Segmentation

Official open-access source code for the MICCAI 2026 paper:

> **FDRAS: Failure Diagnosis and Repair for Airway Segmentation**  
> Accepted at MICCAI 2026.

## Overview

FDRAS is a two-stage failure diagnosis and repair framework for airway segmentation in chest CT. It first predicts complementary SDF-based geometric and tree-level connectivity error maps from an initial airway segmentation and its corresponding CT volume. These predicted failure maps then guide a cross-attention-conditioned residual refinement network to perform spatially targeted corrections. This repository contains the trained checkpoint, inference scripts, training code, and model configurations used in the paper.

## Exvivo dataset

The exvivo airway dataset used in this work is available at:

[https://uoe-my.sharepoint.com/:f:/g/personal/xzhang19_ed_ac_uk/IgCzBhGs_vdxSIZlCZ4d2JEvAY2DnOLe0_hZlbKOF2ju1Ss?e=QlL122](https://uoe-my.sharepoint.com/:f:/g/personal/xzhang19_ed_ac_uk/IgCzBhGs_vdxSIZlCZ4d2JEvAY2DnOLe0_hZlbKOF2ju1Ss?e=QlL122)

## Repository contents

- `checkpoint_best.pt` – trained FDRAS checkpoint used in the paper.
- `predict.py` – single-case inference.
- `infer_dataset.py` – batch inference over a dataset.
- `train.py` – training script.
- `data.py` – data loading and augmentation utilities.
- `model/` – diagnosis and repair network definitions.
- `fdras/fdras_full.yaml` – main training configuration.

## Requirements

- Python >= 3.9
- PyTorch
- MONAI
- nibabel
- numpy
- tqdm

Install the dependencies with your preferred environment manager, for example:

```bash
pip install torch monai nibabel numpy tqdm
```

## Quick start: single-case inference

From the repository root:

```bash
python -m predict \
  --ckpt checkpoint_best.pt \
  --ct /path/to/ct.nii.gz \
  --pred /path/to/preliminary_mask.nii.gz \
  --out_prob output/prob.nii.gz \
  --out_mask output/mask.nii.gz \
  --threshold 0.5
```

## Batch inference

```bash
python -m infer_dataset \
  --ckpt checkpoint_best.pt \
  --data_root /path/to/dataset \
  --ct_subdir imagesTr \
  --pred_subdir nnUNet_masksTr \
  --gt_subdir GT_evalTr \
  --out_root outputs/fdras \
  --threshold 0.5
```

## Training

```bash
python -m train --config fdras/fdras_full.yaml
```

Update the paths in the YAML config to point to your local data before training.

## Notes

Internal reanalysis records and implementation notes are kept under `paper/` for reference.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{fdras2026,
  title={FDRAS: Failure Diagnosis and Repair for Airway Segmentation},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```

The full author list and DOI will be added once the proceedings are published.

## License

This repository is released as open-access source code accompanying the MICCAI 2026 paper. Please refer to the paper for attribution details.
