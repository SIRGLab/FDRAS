# Non‑Adversarial Airway Refinement (No GAN)

Generator‑only refinement network:
- Input: CT + preliminary mask
- Output: refined mask (no discriminator)
- Losses: L1 + Dice + clDice + CCF (configurable)

## Train (ATM, fixed split 60/60)

```bash
python -m runtime_monitoring.repair_model.atr_refine_no_gan.train \
  --config configs/runtime_monitoring/repair_model/atr_refine_no_gan/atr_no_gan_atm.yaml
```

## Inference (single case)

```bash
python -m runtime_monitoring.repair_model.atr_refine_no_gan.predict \
  --ckpt runs/repair_model/atr_refine_no_gan/checkpoints/best.pt \
  --ct runs/CT_Training_Data/ATM_Data/imagesTr/ATM_001_0000.nii.gz \
  --pred runs/CT_Training_Data/ATM_Data/nnUNet_masksTr/ATM_001_0000.nii.gz \
  --out_prob runs/repair_model/atr_refine_no_gan/infer/ATM_001_0000_prob.nii.gz \
  --out_mask runs/repair_model/atr_refine_no_gan/infer/ATM_001_0000_mask.nii.gz \
  --threshold 0.6
```

## Inference (dataset)

```bash
python -m runtime_monitoring.repair_model.atr_refine_no_gan.infer_dataset \
  --ckpt runs/repair_model/atr_refine_no_gan/checkpoints/best.pt \
  --data_root runs/CT_Training_Data/ATM_Data \
  --ct_subdir imagesTr \
  --pred_subdir nnUNet_masksTr \
  --gt_subdir labelsTr \
  --out_root runs/repair_model/atr_refine_no_gan/test_atm \
  --threshold 0.6
```
