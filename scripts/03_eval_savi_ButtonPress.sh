#!/bin/bash

gpu run python src/03_evaluate_savi.py \
  -d experiments/ButtonPress/ \
  --savi_ckpt SAVi_ButtonPress.pth \
  --results_name quant_eval_savi