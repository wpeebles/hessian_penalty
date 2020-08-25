#!/usr/bin/env bash

python train.py --hp_lambda 0.1 --total_kimg 50000 --dataset clevr_simple --nz 12 --num_gpus 4 --warmup_kimg 0
