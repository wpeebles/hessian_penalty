#!/usr/bin/env bash

python train.py --hp_lambda 0 --infogan_lambda 1.0 --infogan_nz 12 --total_kimg 50000 --dataset edges_and_shoes --num_gpus 4
