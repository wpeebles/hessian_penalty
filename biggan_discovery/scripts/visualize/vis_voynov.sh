#!/bin/bash
# The Voynov directions are already orthogonalized, hence the --no_ortho below
python vis_directions.py \
--load_A checkpoints/directions/voynov.pt \
--no_ortho \
--path_size 8 \
--n_samples 8 \
--ndirs 120 \
--fix_class 207 \
--experiment_name voynov_directions \
--parallel --batch_size 96  \
--G_B2 0.999 \
--G_attn 64 \
--G_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 10000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 1 \
--use_multiepoch_sampler \
--resume \
--num_epochs 50