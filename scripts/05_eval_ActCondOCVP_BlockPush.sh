#!/bin/bash


gpu run python src/05_evaluate_OCVP.py \
  -d experiments/BlockPush/ \
  --name_pred_exp ActCondOCVP \
  --savi_ckpt SAVi_BlockPush.pth \
  --pred_ckpt ActCondOCVP_BlockPush.pth \
  --results_name quant_eval_ActCondOCVP \
  --num_seed 6 \
  --num_preds 15 \
  --set_expert_policy

