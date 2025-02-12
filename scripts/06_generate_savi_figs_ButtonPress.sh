#!/bin/bash

gpu run python src/06_generate_figs_savi.py \
  -d experiments/ButtonPress/ \
  --savi_ckpt SAVi_ButtonPress.pth \
  --num_seqs 10 \
  --num_frames 8
