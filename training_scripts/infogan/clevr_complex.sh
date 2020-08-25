#!/usr/bin/env bash

python train.py --hp_lambda 0 --infogan_lambda 1.0 --infogan_nz 10 --total_kimg 50000 --dataset clevr_complex --num_gpus 4
