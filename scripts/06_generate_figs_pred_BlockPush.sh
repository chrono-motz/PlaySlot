#!/bin/bash


gpu run python src/06_generate_figs_pred.py \
  -d experiments/BlockPush/ \
  --name_pred_exp PlaySlot \
  --savi_ckpt SAVi_BlockPush.pth \
  --pred_ckpt PlaySlot_BlockPush.pth \
  --num_seqs 10 \
  --num_seed 1 \
  --num_preds 15 \
  --set_expert_policy

