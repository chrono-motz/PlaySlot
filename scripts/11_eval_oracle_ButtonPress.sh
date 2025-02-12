#!/bin/bash

gpu run python src/11_evaluate_oracle_simulation.py \
  -d experiments/ButtonPress/ \
  --savi_ckpt SAVi_ButtonPress.pth \
  --name_oracle_exp Oracle \
  --oracle_ckpt Oracle_ButtonPress.pth \
  --seed 1000 \
  --num_sims 10 \
  --max_num_steps 20 \
  --save_vis 10
