# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Loss functions."""

import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from training.hessian_penalties import multi_layer_hessian_penalty

# ----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

# ----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.


def G_wgan(G, D, opt, hp_lambda, HP_args, training_set, minibatch_size, lod_in, max_lod, gpu_ix, **kwargs):
    # Note: HP_args.hp_lambda is the MAXIMUM possible loss weighting of the Hessian Penalty (lambda in the paper).
    #       hp_lambda is the CURRENT loss weighting of the Hessian Penalty (lambda_t in the paper)

    # Compute standard WGAN G loss:
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -fake_scores_out

    if HP_args.hp_lambda > 0:  # Compute Hessian Penalty for G:
        hessian_penalty = multi_layer_hessian_penalty(G_z=fake_images_out, z=latents,
                                                      num_rademacher_samples=HP_args.num_rademacher_samples,
                                                      epsilon=HP_args.epsilon, layers_to_reg=HP_args.layers_to_reg,
                                                      current_lod=lod_in, max_lod=max_lod, gpu_ix=gpu_ix)
        # If we're fine-tuning with the Hessian Penalty, we don't want to start computing it until hp_start_nimg:
        hessian_penalty = tf.cond(hp_lambda > 0, lambda: hessian_penalty, lambda: 0.0)
        hessian_penalty = autosummary('Loss/hessian_penalty', hessian_penalty)
    else:
        hessian_penalty = 0.0

    G_loss = tf.reduce_mean(loss) + hp_lambda * hessian_penalty
    return G_loss, hessian_penalty


def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, infogan_nz,
              wgan_lambda=10.0,     # Weight for the gradient penalty term.
              wgan_epsilon=0.001,    # Weight for the epsilon term, \epsilon_{drift}.
              wgan_target=1.0,      # Target value for gradient magnitudes.
              gpu_ix=None):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))

    if infogan_nz > 0:
        with tf.name_scope('InfoGANLoss'):
            ops = fake_scores_out.graph.get_operations()

            def filter_fn(op):  # Very similar to the corresponding function in hessian_penalties.py
                this_layer = 'QEncoding' in op.name
                this_gpu = 'GPU%d' % gpu_ix in op.name
                this_model = 'D_loss' in op.name
                op_found = this_layer and this_gpu and this_model
                return op_found

            r_op = list(filter(filter_fn, ops))
            for r in r_op:
                print('Using %s' % r.name)
            assert len(r_op) == 1, 'Found %s ops with name QEncoding' % len(r_op)
            encoding = fake_scores_out.graph.get_tensor_by_name('%s:0' % r_op[0].name)
            print('Regularizing first %s Z components with InfoGAN Loss' % infogan_nz)
            mutual_information_loss = tf.losses.mean_squared_error(latents[:, :infogan_nz], encoding)
            mutual_information_loss = autosummary('Loss/InfoGAN', mutual_information_loss)
    else:
        mutual_information_loss = 0.0

    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss, mutual_information_loss

# ----------------------------------------------------------------------------
