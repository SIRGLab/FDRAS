# FDRAS diagnostic error-map AUROC notes (AeroPath)

> Per MICCAI policy, the open-access source package must not introduce new experiments, training, or data. The AUROC values below are derived solely from the archived diagnostic error-map evaluation outputs generated for the submitted manuscript.

## Summary

Diagnostic error-map quality is measured by strict-XOR AUROC: the ability of each error map to rank voxels inside the symmetric difference between the preliminary prediction and the ground-truth mask higher than voxels outside it. Two preliminary baselines are evaluated: nnUNet and TfeNet.

| Error map | Baseline | n | Mean AUROC | Range |
|---|---|---:|---:|---|
| Tree-error map | nnUNet | 27 | 0.6803 | 0.5147 – 0.8264 |
| Tree-error map | TfeNet | 27 | 0.6416 | 0.4969 – 0.7926 |
| SDF-error map  | nnUNet | 27 | 0.4812 | 0.4532 – 0.5185 |
| SDF-error map  | TfeNet | 27 | 0.4684 | 0.4406 – 0.5353 |

## Interpretation

- The **tree-error map** carries weak but consistent discriminative signal for both nnUNet and TfeNet preliminary masks (mean AUROC > 0.5).
- The **SDF-error map** is close to chance for both baselines, indicating it is not a reliable standalone failure predictor on this dataset.
- These values inform the ablation results: the repair network relies on the tree-error map for content, while the SDF map primarily acts as a spatial regularizer rather than a strong diagnostic signal.
