## The Hessian Penalty &mdash; Official Implementation

[Paper](https://arxiv.org/abs/2008.10599) | [Project Page](https://www.wpeebles.com/hessian-penalty) | [ECCV 2020 Spotlight Video](https://youtu.be/uZyIcTkSSXA) | [The Hessian Penalty in 90 Seconds](https://youtu.be/jPl-0EN6S1w)

Home | [PyTorch BigGAN Discovery](biggan_discovery) | [TensorFlow ProGAN Regularization](progan_experiments)

![Teaser image](teaser_images/teaser_small.gif)

This repo contains code for our new regularization term that encourages disentanglement in neural networks. It efficiently optimizes the Hessian of your neural network to be diagonal in an input, leading to disentanglement in that input. We showcase its usage in generative adversarial networks (GANs), but you can use our code for other neural networks too.

[**The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement**](https://www.wpeebles.com/hessian-penalty)<br>
[William Peebles](https://www.wpeebles.com/), [John Peebles](http://johnpeebles.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Alexei Efros](https://people.eecs.berkeley.edu/~efros/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/)<br>
UC Berkeley, Yale, Adobe Research, MIT CSAIL<br>
ECCV 2020 (Spotlight)

This repo contains the following:

* Portable Hessian Penalty implementations in both PyTorch and TensorFlow
* [Edges+Shoes and CLEVR ProGAN Experiments in TensorFlow](progan_experiments)
* :fire: **NEW January 2021** :fire: [BigGAN Direction Discovery Experiments in PyTorch](biggan_discovery)

Below are some examples of latent space directions and components learned via the Hessian Penalty:

![Dogs teaser](teaser_images/dogs_ours.gif)
![Dogs background teaser](teaser_images/dogs_bg.gif)
![Dogs lighting teaser](teaser_images/dogs_light.gif)
![Church teaser](teaser_images/church_colorize.gif)
![CLEVR teaser](teaser_images/clevr.gif)

## Adding the Hessian Penalty to Your Code

We provide portable implementations of the Hessian Penalty that you can easily add to your projects.

* PyTorch: [`hessian_penalty_pytorch.py`](hessian_penalty_pytorch.py)

* TensorFlow: [`hessian_penalty_tf.py`](hessian_penalty_tf.py) (needs `pip install tensorflow-probability`)

* NumPy (easiest to read): [`hessian_penalty_np.py`](hessian_penalty_np.py)

Adding the Hessian Penalty to your own code is very simple:

```python
from hessian_penalty_pytorch import hessian_penalty

net = MyNeuralNet()
input = sample_input()
loss = hessian_penalty(G=net, z=input)
loss.backward()
```

See our [Tips and Tricks](tips_and_tricks.md) for some advice about training with the Hessian Penalty and avoiding pitfalls. Our code supports regularizing multiple activations simultaneously; see the fourth bullet point in Tips and Tricks for how to enable this feature.

## Getting Started

This section and below are only needed if you want to visualize/evaluate/train with our code and models. For using the Hessian Penalty in your own code, you can copy one of the files mentioned in the above section.

Both the TensorFlow and PyTorch codebases are tested with Linux on NVIDIA GPUs. You need at least Python 3.6. To get started, download this repo:

```bash
git clone git@github.com:wpeebles/hessian_penalty.git
cd hessian_penalty
```

Then, set-up your environment. You can use the [`environment.yml`](environment.yml) file to set-up a [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

```bash
conda env create -f environment.yml
conda activate hessian
```

If you opt to use your environment, we recommend using TensorFlow 1.14.0 and PyTorch >= 1.6.0. Now you're all set-up.

## [TensorFlow ProgressiveGAN Regularization Experiments](progan_experiments)

## [PyTorch BigGAN Direction Discovery Experiments](biggan_discovery)

## Citation

If our code aided your research, please cite our [paper](https://arxiv.org/pdf/2008.10599.pdf):
```
@inproceedings{peebles2020hessian,
  title={The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement},
  author={Peebles, William and Peebles, John and Zhu, Jun-Yan and Efros, Alexei A. and Torralba, Antonio},
  booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Acknowledgments

We thank Pieter Abbeel, Taesung Park, Richard Zhang, Mathieu Aubry, Ilija Radosavovic, Tim Brooks, Karttikeya Mangalam, and all of BAIR for valuable discussions and encouragement. This work was supported, in part, by grants from SAP, Adobe, and Berkeley DeepDrive.
