"""
An implementation of the Hessian Penalty in pure NumPy (8 lines of code).
PyTorch Implementation (GPU + Multi-Layer): hessian_penalty_pytorch.py
TensorFlow Implementation (GPU + Multi-Layer): hessian_penalty_tf.py
Hessian Penalty Paper: https://arxiv.org/pdf/2008.10599.pdf
"""

import numpy as np


def hessian_penalty(G, z, k=2, epsilon=0.1, reduction=np.max, G_z=None, **G_kwargs):
    """
    Official NumPy Hessian Penalty implementation (single-layer).

    :param G: Function that maps input z to NumPy array
    :param z: Input to G that the Hessian Penalty will be computed with respect to
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>

    :return: A differentiable scalar (the hessian penalty)
    """
    if G_z is None:
        G_z = G(z, **G_kwargs)
    xs = np.random.choice([-epsilon, epsilon], size=[k, *z.shape], replace=True)  # Sample from Rademacher distribution
    second_orders = [G(z + x, **G_kwargs) - 2 * G_z + G(z - x, **G_kwargs) for x in xs]
    second_orders = np.stack(second_orders) / (epsilon ** 2)  # Shape = (k, *G(z).shape)
    per_neuron_loss = np.var(second_orders, axis=0, ddof=1)  # Compute unbiased variance over k Hessian directions
    loss = reduction(per_neuron_loss)
    return loss


def _test_hessian_penalty():
    """
    A simple test to verify the implementation.
    Function: G(z) = z_0**2 * z_1
    Ground Truth Hessian Penalty: 16 * z_0**2
    """
    batch_size = 10
    nz = 2
    z = np.random.randn(batch_size, nz)
    def reduction(x): return np.abs(x).mean()
    def G(z): return (z[:, 0] ** 2) * z[:, 1]
    ground_truth = reduction(16 * z[:, 0] ** 2)
    # In this simple example, we use k=100 to reduce variance, but when applied to neural networks
    # you will probably want to use a small k (e.g., k=2) due to memory considerations.
    predicted = hessian_penalty(G, z, G_z=None, k=100, reduction=reduction)
    print('Ground Truth: %s' % ground_truth)
    print('Approximation: %s' % predicted)  # This should be close to ground_truth, but not exactly correct
    print('Difference: %s' % (str(100 * abs(predicted - ground_truth) / ground_truth) + '%'))


if __name__ == '__main__':
    _test_hessian_penalty()
