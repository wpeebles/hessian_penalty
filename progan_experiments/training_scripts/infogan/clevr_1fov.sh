#!/usr/bin/env bash

python train.py --hp_lambda 0 --infogan_lambda 1.0 --infogan_nz 1 --total_kimg 50000 --dataset clevr_1fov --num_gpus 4
