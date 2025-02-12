#!/bin/bash


gpu run python src/05_evaluate_OCVP.py \
  -d experiments/ButtonPress/ \
  --name_pred_exp ActCondOCVP \
  --savi_ckpt SAVi_ButtonPress.pth \
  --pred_ckpt ActCondOCVP_ButtonPress.pth \
  --results_name quant_eval_ActCondOCVP \
  --num_seed 6 \
  --num_preds 15 \
  --set_expert_policy

