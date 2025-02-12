#!/bin/bash


gpu run python src/06_generate_figs_pred.py \
  -d experiments/ButtonPress/ \
  --name_pred_exp PlaySlot \
  --savi_ckpt SAVi_ButtonPress.pth \
  --pred_ckpt PlaySlot_ButtonPress.pth \
  --num_seqs 10 \
  --num_seed 1 \
  --num_preds 15 \
  --set_expert_policy

