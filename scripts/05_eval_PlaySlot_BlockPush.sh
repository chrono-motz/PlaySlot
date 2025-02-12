#!/bin/bash


gpu run python src/05_evaluate_PlaySlot.py \
  -d experiments/BlockPush/ \
  --name_pred_exp PlaySlot \
  --savi_ckpt SAVi_BlockPush.pth \
  --pred_ckpt PlaySlot_BlockPush.pth \
  --results_name quant_eval_playslot \
  --post_only \
  --num_seed 6 \
  --num_preds 15 \
  --set_expert_policy

