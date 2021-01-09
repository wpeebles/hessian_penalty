"""
Various minimal tests to sanity-check the implementation of multi_layer_hessian_penalty.
You can run these tests on CPU or GPU (the "G_loss/GPU0" scope is just to spoof the naming expected by multi_layer_hessian_penalty).
Due to the stochastic nature of the Hessian Penalty approximation, the computation won't exactly match the ground truth.
"""
import numpy as np
from training.hessian_penalties import multi_layer_hessian_penalty
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def reset_tf(seed=500):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def zero_hessian_penalty_test():
    """
    Input:
        scalar x
    Function:
        G(x) = x ** 2
    True Hessian Penalty:
        0.0 (since x is a scalar, there are no off-diagonal elements in G(x)'s Hessian)
    """
    reset_tf()
    x = tf.placeholder(shape=[], dtype=tf.float32)
    with tf.name_scope('G_loss/GPU0'):
        x_sqr = tf.identity(x ** 2, name='1x1/x_sqr')
    hp_tf = multi_layer_hessian_penalty(G_z=x_sqr, z=x, num_rademacher_samples=4, layers_to_reg=['1x1/x_sqr'],
                                        current_lod=0.0, max_lod=1.0, gpu_ix=0, log=False)

    sess = tf.Session()
    hp_np = sess.run(hp_tf, feed_dict={x: np.random.randn()})
    print(f'Hessian Penalty: {hp_np} | Ground Truth: 0.0')


def quadratic_hessian_penalty_test():
    """
    Input:
        x with shape (1, 2) (a two-dimensional vector with components x0 and x1)
    Function:
        G(x) = x0 * x1
    True Hessian Penalty:
        (1 + 1) ** 2 = 4.0
    """
    reset_tf()
    x = tf.placeholder(shape=[1, 2], dtype=tf.float32)
    with tf.name_scope('G_loss/GPU0'):
        x_quad = tf.identity(x[:, 0] * x[:, 1], name='1x1/x_quad')
    hp_tf = multi_layer_hessian_penalty(G_z=x_quad, z=x, num_rademacher_samples=8, layers_to_reg=['1x1/x_quad'],
                                        current_lod=0.0, max_lod=1.0, gpu_ix=0, log=False)

    sess = tf.Session()
    hp_np = sess.run(hp_tf, feed_dict={x: np.random.randn(1, 2)})
    print(f'Hessian Penalty: {hp_np} | Ground Truth: 4.0')


def order_4_hessian_penalty_test():
    """
    Input:
        x with shape (1, 2) (a two-dimensional vector with components x0 and x1)
    Function:
        G(x) = 0.25 * (x0 ** 2) * (x1 ** 2)
    True Hessian Penalty:
        (2 * x0 * x1) ** 2
    """
    reset_tf()
    x = tf.placeholder(shape=[1, 2], dtype=tf.float32)
    with tf.name_scope('G_loss/GPU0'):
        x_order_4 = tf.identity(0.25 * (x[:, 0] ** 2) * (x[:, 1] ** 2), name='1x1/x_order_4')
    hp_tf = multi_layer_hessian_penalty(G_z=x_order_4, z=x, num_rademacher_samples=16, layers_to_reg=['1x1/x_order_4'],
                                        current_lod=0.0, max_lod=1.0, gpu_ix=0, log=False)

    sess = tf.Session()
    x_in = np.random.randn(1, 2)
    gt = (2 * x_in[:, 0] * x_in[:, 1])[0] ** 2
    hp_np = sess.run(hp_tf, feed_dict={x: x_in})
    print(f'Hessian Penalty: {hp_np} | Ground Truth: {gt}')


def batched_order_4_hessian_penalty_test():
    """
    Input:
        x with shape (4, 2) (batch of two-dimensional vectors)
    Function:
        G(x) = 0.25 * (x0 ** 2) * (x1 ** 2)
    True Hessian Penalty (using max reduction):
        max_{x0,x1} [(2 * x0 * x1) ** 2]
    """
    reset_tf()
    x = tf.placeholder(shape=[4, 2], dtype=tf.float32)
    with tf.name_scope('G_loss/GPU0'):
        x_order_4 = tf.identity(0.25 * (x[:, 0] ** 2) * (x[:, 1] ** 2), name='1x1/x_order_4')
    hp_tf = multi_layer_hessian_penalty(G_z=x_order_4, z=x, num_rademacher_samples=32, layers_to_reg=['1x1/x_order_4'],
                                        current_lod=0.0, max_lod=1.0, gpu_ix=0, log=False)

    sess = tf.Session()
    x_in = np.random.randn(4, 2)
    gt = np.max((2 * x_in[:, 0] * x_in[:, 1]) ** 2, axis=0)
    hp_np = sess.run(hp_tf, feed_dict={x: x_in})
    print(f'Hessian Penalty: {hp_np} | Ground Truth: {gt}')


def multi_layer_polynomial_hessian_penalty_test():
    """
    Input:
        x with shape (4, 2) (batch of two-dimensional vectors)
    Function:
        (Layer 1) G(x) = 0.25 * (x0 ** 2) * (x1 ** 2)
        (Layer 2) F(x) = 10 * G(x)
    True Hessian Penalty (using mean reduction):
        (Layer 1) mean_{x0,x1} [(2 * x0 * x1) ** 2]
        (Layer 2) mean_{x0,x1} [10 * (2 * x0 * x1)] ** 2
        Returned Hessian Penalty = Layer 1 Hessian Penalty + Layer 2 Hessian Penalty
    """
    reset_tf()
    x = tf.placeholder(shape=[4, 2], dtype=tf.float32)
    with tf.name_scope('G_loss/GPU0'):
        x_order_4 = tf.identity(0.25 * (x[:, 0] ** 2) * (x[:, 1] ** 2), name='1x1/x_order_4')
        x_mul = tf.identity(10 * x_order_4, name='1x1/x_mul')
    hp_tf = multi_layer_hessian_penalty(G_z=x_mul, z=x, num_rademacher_samples=32,
                                        layers_to_reg=['1x1/x_order_4', '1x1/x_mul'],
                                        current_lod=0.0, max_lod=1.0, gpu_ix=0, log=False,
                                        reduction=tf.reduce_mean)

    sess = tf.Session()
    x_in = np.random.randn(4, 2)
    first_layer_hp_before_squaring = (2 * x_in[:, 0] * x_in[:, 1])
    gt = np.mean(first_layer_hp_before_squaring ** 2) + np.mean(100 * first_layer_hp_before_squaring ** 2)
    hp_np = sess.run(hp_tf, feed_dict={x: x_in})
    print(f'Hessian Penalty: {hp_np} | Ground Truth: {gt}')


if __name__ == '__main__':
    zero_hessian_penalty_test()
    quadratic_hessian_penalty_test()
    order_4_hessian_penalty_test()
    batched_order_4_hessian_penalty_test()
    multi_layer_polynomial_hessian_penalty_test()
