#!/bin/bash

gpu run python src/06_generate_figs_savi.py \
  -d experiments/BlockPush/ \
  --savi_ckpt SAVi_BlockPush.pth \
  --num_seqs 10 \
  --num_frames 8
