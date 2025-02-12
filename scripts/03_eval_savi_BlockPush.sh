#!/bin/bash

gpu run python src/03_evaluate_savi.py \
  -d experiments/BlockPush/ \
  --savi_ckpt SAVi_BlockPush.pth \
  --results_name quant_eval_savi