#!/bin/bash


gpu run python src/06_generate_figs_pred.py \
  -d experiments/Sketchy/ \
  --name_pred_exp PlaySlot \
  --savi_ckpt SAVi_Sketchy.pth \
  --pred_ckpt PlaySlot_Sketchy.pth \
  --num_seqs 10 \
  --num_seed 1 \
  --num_preds 15

