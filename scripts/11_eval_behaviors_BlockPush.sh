#!/bin/bash

gpu run python src/11_evaluate_behavior_on_simulation.py \
  -d experiments/BlockPush/ \
  --savi_ckpt SAVi_BlockPush.pth \
  --name_pred_exp PlaySlot \
  --pred_ckpt PlaySlot_BlockPush.pth \
  --name_beh_exp Policy_AllDemos \
  --beh_ckpt Policy_checkpoint_epoch_500.pth \
  --seed 1000 \
  --num_sims 10 \
  --max_num_steps 20 \
  --save_vis 10
