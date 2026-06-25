# Supplementary table: archived TfeNet–AeroPath reanalysis

> Per MICCAI policy, the open-access source package must not introduce new experiments, training, or data. All results reported below are therefore derived solely from reanalysis of the archived predictions and evaluation outputs generated for the submitted manuscript.

## (a) TfeNet → TfeNet+FDRAS on AeroPath (*n* = 27)

| Metric | TfeNet | +FDRAS | Δ (pp) | Paired *p* | Improved cases |
|---|---:|---:|---:|---|---:|
| DSC | 86.37 | 83.78 | −2.59 | 0.020 | 12/27 |
| Pre | 89.05 | 78.89 | −10.16 | &lt;0.001 | 0/27 |
| TD  | 91.47 | 92.38 | +0.91 | &lt;0.001 | 25/27 |
| BD  | 88.53 | 89.81 | +1.28 | &lt;0.001 | 19/27 |

## (b) Error-map ablation on AeroPath

| Setting | DSC | TD | BD | Pre |
|---|---:|---:|---:|---:|
| Full         | 83.70 | 92.30 | 89.80 | 78.80 |
| w/o tree map | 85.18 | 92.14 | 89.31 | 81.89 |
| w/o SDF map  | 79.55 | 93.04 | 89.83 | 71.79 |

FDRAS improves TD and BD on TfeNet–AeroPath but reduces DSC and Pre. The error-map analysis shows complementary effects: removing the tree map improves DSC and Pre with slightly reduced TD and BD, whereas removing the SDF map slightly improves TD and BD but substantially reduces DSC and Pre. The full model therefore provides the most balanced performance across voxel-level accuracy and branch-level structural coverage.
