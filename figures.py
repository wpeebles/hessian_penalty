"""
Code for generating some of the figures/numbers from our paper.

When using --fig active (activeness scores), you need to add --model_names and --dataset_names. For example, the
following command works:

python figures.py --fig active --models clevr_1fov_fs clevr_1fov_bl --model_names HP BL --dataset_names CLEVR-1FOV
"""

from dnnlib import EasyDict
import dnnlib.tflib as tflib
import argparse
import os
import config
from training.misc import load_pkl, find_pkl
from training.vis_tools import normalize_latents, prepro_imgs
from hessian_penalty_np import hessian_penalty
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from glob import glob
from download import find_model


def main():
    parser = argparse.ArgumentParser(description='Visualize the Disentanglement of ProgressiveGAN w/ Hessian Penalty')

    # Model/Dataset Parameters:
    parser.add_argument('--models', required=True, nargs='+',
                        help='Number of experiment in results directory. You can specify multiple experiments '
                             'to generate visuals for all of them with one call to this script.')
    parser.add_argument('--fig', type=str, required=True, choices=['hessians', 'heatmaps', 'diag', 'active'],
                        help='Which figure to synthesize?')
    parser.add_argument('--snapshot_kimgs', default=['latest'], nargs='+',
                        help='network-snapshot-<snapshot_kimg>.pkl to evaluate. This should either be "latest" or '
                             'a list of length equal to models (each model needs a snapshot_kimg).')
    parser.add_argument('--z', type=int, default=None, nargs='+',
                        help='If specified, this script will only generate videos for the specified z components. '
                             'Otherwise, by default it will generate videos for all z components.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Seed for sampling the latent noise')
    parser.add_argument('--nz_per_vid', default=12, type=int,
                        help='Number of z components to visualize per-video. This controls the "height" of the '
                             'generated videos.')
    parser.add_argument('--samples', default=8, type=int,
                        help='Number of z samples to use (=interp_batch_size). This controls the "width" of the '
                             'generated videos.')
    parser.add_argument('--extent', default=2.0, type=float,
                        help='How "far" to move the z components (from -extent to +extent)')
    parser.add_argument('--minibatch_size', default=100, type=int,
                        help='Batch size to use when generating frames. If you get memory errors, try decreasing this.')
    parser.add_argument('--interpolate_pre_norm', action='store_true', default=False,
                        help='If specified, interpolations are performed before the first pixel norm layer in G. '
                             'You should use this when nz is small (e.g., CLEVR-U).')
    parser.add_argument('--model_names', default=None, nargs='+',
                        help='Give names to the models you are evaluating (used when --fig 4).')
    parser.add_argument('--dataset_names', default=None, nargs='+',
                        help='Give names to the datasets you are evaluating (used when --fig 4).')

    opt = parser.parse_args()
    opt = EasyDict(vars(opt))

    for model in opt.models:
        find_model(model)

    if os.path.isdir(opt.models[0]):
        opt.models = sorted(glob(f'{opt.models[0]}/*/*.pkl'))

    if len(opt.snapshot_kimgs) == 1 and len(opt.models) > 1:
        assert opt.snapshot_kimgs[0] == 'latest'
        opt.snapshot_kimgs = ['latest'] * len(opt.models)
    else:
        assert len(opt.snapshot_kimgs) == len(opt.models)
    for seed in opt.seeds:
        opt.seed = seed
        run(opt)


###########################################
# Helper functions for generating figures #
###########################################


def get_Gs(opt):
    # Find and load the network checkpoints, 1-by-1:
    for exp_number, snapshot_kimg in zip(opt.models, opt.snapshot_kimgs):
        resume_pkl = find_model(exp_number)
        if not resume_pkl:
            if not exp_number.endswith('.pkl'):  # Look for a pkl in results directory
                results_dir = os.path.join(os.getcwd(), config.result_dir)
                resume_pkl = find_pkl(results_dir, int(exp_number), snapshot_kimg)
            else:
                resume_pkl = exp_number
        tflib.init_tf()
        _, _, _Gs = load_pkl(resume_pkl)
        nz = _Gs.input_shapes[0][1]
        Gs = tflib.Network(name='Gs', func_name='training.networks_progan.G_paper',
                           latent_size=nz, num_channels=3, resolution=128, label_size=0)
        Gs.copy_vars_from(_Gs)
        print(f'Visualizing pkl: {resume_pkl} with seed={opt.seed}')
        if nz < 12 and not opt.interpolate_pre_norm:
            print(f'Model {exp_number} uses a small z vector (nz={nz}); you might want to add '
                  f'--interpolate_pre_norm to your command.')
        yield Gs, nz


def get_names(opt):
    checkpoints = []
    for exp_number, snapshot_kimg in zip(opt.models, opt.snapshot_kimgs):
        if os.path.exists(os.path.join('pretrained_models', f'{exp_number}.pkl')):
            checkpoint_name = exp_number
        elif not exp_number.endswith('.pkl'):
            results_dir = os.path.join(os.getcwd(), config.result_dir)
            resume_pkl = find_pkl(results_dir, exp_number, snapshot_kimg)
            checkpoint_name = resume_pkl.split('network-snapshot-')[-1].split('.pkl')[0]
            # exp_name = resume_pkl.split('/')[-2]
        else:
            checkpoint_name = exp_number.split('/')[-1].split('.pkl')[0]
        checkpoints.append(checkpoint_name)
    return '_'.join(checkpoints)


def sample(nz, opt):
    np.random.seed(opt.seed)
    z = np.random.randn(opt.samples, nz)
    if not opt.interpolate_pre_norm:
        z = normalize_latents(z)
    return z


def full_hessian(G, z, epsilon=0.1, nc=3, h=128, w=128):
    """
    Computes a second-order centered finite difference approximation of H_z[G(z)].
    G: function z --> x
    z: rank-2 tensor (N, |z|)
    """
    N, nz = z.shape
    eps = np.eye(nz) * epsilon
    H = np.zeros((N, nc, h, w, nz, nz))
    for i in range(nz):
        di = eps[i][np.newaxis]
        for j in range(nz):
            if i >= j:  # Hessian is symmetric
                dj = eps[j][np.newaxis]
                H_ij = (G(z + di + dj) - G(z + di - dj) - G(z - di + dj) + G(z - di - dj)) / (4 * epsilon ** 2)
                H[:, :, :, :, i, j] = H_ij
                H[:, :, :, :, j, i] = H_ij
    return H


def hessian_of_pixel(Gs, nz, opt, choose_by='largest_hp'):
    def Gs_np(z): return Gs.run(z, None, is_validation=True, normalize_latents=opt.interpolate_pre_norm)
    z = sample(nz, opt)
    H = full_hessian(Gs_np, z)  # (N, C, H, W, |z|, |z|)

    # Pick individual Hessian matrices to return:
    if choose_by == 'largest_hp':  # Visualize the Hessian of the pixel with the largest Hessian Penalty for each image
        def identity(x): return x
        hessian_penalties = hessian_penalty(Gs_np, z, k=200, reduction=identity)  # (N, C, H, W)
        H_ixs = hessian_penalties.reshape((opt.samples, -1)).argmax(axis=1)  # (N,)
    elif choose_by == 'smallest_hp':  # Visualize the Hessian of the pixel with the smallest Hessian Penalty per image:
        def identity(x): return x
        hessian_penalties = hessian_penalty(Gs_np, z, k=200, reduction=identity)  # (N, C, H, W)
        H_ixs = hessian_penalties.reshape((opt.samples, -1)).argmin(axis=1)  # (N,)
    elif choose_by == 'random':  # Visualize the Hessian of a random pixel in each sample
        H_ixs = np.random.choice(np.prod(H.shape[1:4]), size=(opt.samples,))  # (N,)
    elif choose_by == 'all':  # Return ALL Hessian matrices for every pixel
        return H
    else:
        raise NotImplementedError

    nz = H.shape[-1]
    H = H.reshape((opt.samples, -1, nz, nz))  # (N, C*H*W, nz, nz)
    batch_ixs = np.arange(opt.samples)  # (N,)
    hessians = np.take(H, H_ixs, axis=1)[batch_ixs, batch_ixs]  # (N, nz, nz)
    return hessians


def diagonalness(opt):
    # This functions computes the fraction of Hessians whose largest element
    # lies on the diagonal of the matrix. We do this by computing the max value
    # in each Hessian, then creating an indicator matrix which is 1 where the max
    # value occurs and 0 elsewhere. Then, we mask-out the off-diagonal entries of
    # this matrix and see if there is a 1 anywhere in the matrix. We count how many
    # Hessians satisfy this test. We also compute the relative value of diagonal entries
    # on the Hessian to the offdiagonal entries.
    for i, (G, nz) in enumerate(get_Gs(opt)):  # Iterate over all G networks we want to evaluate:
        Hs = np.abs(hessian_of_pixel(G, nz, opt, choose_by='all'))  # (N, C, H, W, |z|, |z|)
        Hs = Hs.reshape((-1, nz, nz))  # (N*C*H*W, |z|, |z|)
        max_val = Hs.reshape((-1, nz * nz)).max(axis=1)  # (N*C*H*W,)
        Hs_max = (Hs == max_val[:, np.newaxis, np.newaxis])  # (N*C*H*W, 1, 1)
        I = np.eye(nz)[np.newaxis]  # (1, |z|, |z|)
        strong_diagonal = np.any((Hs_max * I).reshape((-1, nz * nz)), axis=1).mean()  # (1,)
        print(f'[G{i}] Fraction of Hessians with Max on Diagonal: {strong_diagonal}')

        diagonal_sum = (Hs * I).sum(axis=(1, 2)) / nz  # (N,)
        offdiag_sum = (Hs * (1 - I)).sum(axis=(1, 2)) / (nz * (nz - 1))  # (N,)
        ratio = (diagonal_sum / (offdiag_sum + 1e-8)).mean()  # (1,)
        print(f'[G{i}] Relative Diagonal Strength: {ratio}')


def activeness(opt, extent=2):
    scores = [[] for _ in range(len(opt.models))]
    for i, (Gs, nz) in enumerate(get_Gs(opt)):
        z = sample(nz, opt)
        for z_i in tqdm(range(nz)):
            z1, z2, z3 = deepcopy(z), deepcopy(z), deepcopy(z)
            z1[:, z_i] = -extent
            z2[:, z_i] = 0
            z3[:, z_i] = extent
            Gz1 = Gs.run(z1, None, is_validation=True, normalize_latents=opt.interpolate_pre_norm)
            Gz2 = Gs.run(z2, None, is_validation=True, normalize_latents=opt.interpolate_pre_norm)
            Gz3 = Gs.run(z3, None, is_validation=True, normalize_latents=opt.interpolate_pre_norm)
            Gzs = np.stack([Gz1, Gz2, Gz3])
            score = np.var(Gzs, axis=0, ddof=1).mean()
            scores[i].append(score)
    return scores


###########################################
#     Functions for generating figures    #
###########################################


def hessian_comparison(opt, out_path, choose_by='largest_hp'):
    hessians_for_each_G = []
    for G, nz in get_Gs(opt):
        hessians_for_each_G.extend(np.abs(hessian_of_pixel(G, nz, opt, choose_by)))  # Absolute value the Hessians
    n_Gs = len(opt.models)

    H_concat = np.stack(hessians_for_each_G)
    vmin, vmax = H_concat.min(), H_concat.max()

    outer = gridspec.GridSpec(1, 2, width_ratios=[9, 1])
    inner = gridspec.GridSpecFromSubplotSpec(n_Gs, opt.samples, subplot_spec=outer[0])  # Each row corresponds to one G

    fig = plt.figure(figsize=(7, 3.5))
    axs = [plt.Subplot(fig, inner[i]) for i in range(n_Gs * opt.samples)]
    cbar_ax = fig.add_axes([.8, .2, .025, .6])
    cbar_ticks = np.linspace(vmin, vmax, num=6)
    for i, ax in enumerate(axs):
        sns.heatmap(hessians_for_each_G[i], ax=ax, vmax=vmax, vmin=vmin, square=True,
                    cbar=not i, xticklabels=False, yticklabels=False, cmap='YlGnBu', cbar_ax=cbar_ax,
                    cbar_kws=dict(ticks=cbar_ticks))
        fig.add_subplot(ax)

    plt.savefig(os.path.join(out_path, f'hessians_{choose_by}.pdf'))


def hessian_penalty_heatmap(opt, out_path, alpha=0.08):
    matplotlib.rcParams['figure.dpi'] = 500
    heatmaps = []
    images = []
    for Gs, nz in get_Gs(opt):
        z = sample(nz, opt)
        G_z = Gs.run(z, None, is_validation=True, normalize_latents=opt.interpolate_pre_norm)
        def reduction(x): return x.max(axis=1)
        def Gs_np(z): return Gs.run(z, None, is_validation=True, normalize_latents=opt.interpolate_pre_norm)
        hessian_penalties = hessian_penalty(Gs_np, z, k=2, reduction=reduction)  # (N, H, W)
        heatmaps.append(hessian_penalties)
        images.append(G_z)
    heatmaps = np.concatenate(heatmaps, axis=0)
    images = prepro_imgs(np.concatenate(images, axis=0))
    n_Gs = len(opt.models)

    vmin, vmax = heatmaps.min(), heatmaps.max()

    outer = gridspec.GridSpec(1, 2, width_ratios=[9, 1])
    inner = gridspec.GridSpecFromSubplotSpec(n_Gs, opt.samples, subplot_spec=outer[0])  # Each row corresponds to one G

    fig = plt.figure(figsize=(7, 3.5))
    axs = [plt.Subplot(fig, inner[i]) for i in range(n_Gs * opt.samples)]
    cbar_ax = fig.add_axes([.8, .2, .025, .6])
    cbar_ticks = np.linspace(vmin, vmax, num=6)
    for i, ax in enumerate(axs):
        hmap = sns.heatmap(heatmaps[i], ax=ax, vmax=vmax, vmin=vmin, square=True,
                           cbar=not i, xticklabels=False, yticklabels=False, cmap='YlGnBu',
                           cbar_ax=cbar_ax, cbar_kws=dict(ticks=cbar_ticks), zorder=0)
        # Downscale images[i] to the resolution of heatmaps[i] if needed:
        img = np.asarray(Image.fromarray(images[i]).resize((heatmaps.shape[-1], heatmaps.shape[-2]), Image.ANTIALIAS))
        hmap.imshow(img, zorder=1, alpha=alpha)
        fig.add_subplot(ax)
    plt.savefig(os.path.join(out_path, f'heatmaps.png'))  # Using .pdf instead produces very "laggy" figures


def activeness_histogram(opt, out_path, model_names=None, dataset_names=None, extent=2, sort_type='per_model'):
    n_models = len(model_names)
    n_datasets = len(dataset_names)
    assert n_datasets * n_models == len(opt.models), 'You need to specify --model_names and --dataset_names'
    assert sort_type in ['per_model', 'shared']

    # Each subplot corresponds to a single dataset:
    fig = make_subplots(rows=int(np.ceil(n_datasets / 2)), cols=2, subplot_titles=dataset_names)
    scores = activeness(opt, extent)
    colors = px.colors.qualitative.Pastel[:n_models]

    for dset_ix in range(n_datasets):
        # Sort the first method's z components by activeness and use that ranking for others:
        if sort_type == 'shared':
            global_ranks = np.argsort(scores[0])[::-1]
            global_z_components = [f'z{z_ix:02}' for z_ix in global_ranks]
            global_ranks = [global_ranks] * n_models
            global_z_components = [global_z_components] * n_models
        else:  # Otherwise, the order of z components will be determined per-method
            global_ranks = None
            global_z_components = None

        for model_ix in range(n_models):
            G_scores = np.asarray(scores.pop(0))
            if sort_type == 'shared':
                G_scores = G_scores[global_ranks]
                z_components = global_z_components
            else:
                ranks = np.argsort(G_scores)[::-1]
                print(ranks)
                z_components = None
                G_scores = G_scores[ranks]
            graph = go.Bar(x=z_components, y=G_scores, name=model_names[model_ix],
                           marker_color=colors[model_ix], showlegend=not dset_ix)
            fig.add_trace(graph, row=1 + dset_ix // 2, col=1 + dset_ix % 2)  # row and col are 1-indexed

    fig.update_xaxes(title_text='Z Component', titlefont=dict(size=18))
    fig.update_yaxes(title_text='Activeness', titlefont=dict(size=18))
    fig.update_layout(font_size=18)
    for i in fig['layout']['annotations']:  # https://github.com/plotly/plotly.py/issues/985
        i['font']['size'] = 26

    plotly.offline.plot(fig, filename=os.path.join(out_path, 'activeness.html'))


def run(opt):
    # Create directory for saving figures (hessian_penalty/visuals/figure_{fig}/seed_{N}):
    if opt.fig != 'diag':
        group_name = get_names(opt)
        out_path = os.path.join(config.vis_dir, f'figure_{opt.fig}', group_name, f'seed_{opt.seed}')
        os.makedirs(out_path, exist_ok=True)

    # Generate and save the figure:
    if opt.fig == 'hessians':
        hessian_comparison(opt, out_path)
    elif opt.fig == 'diag':
        diagonalness(opt)
    elif opt.fig == 'heatmaps':
        hessian_penalty_heatmap(opt, out_path)
    elif opt.fig == 'active':
        activeness_histogram(opt, out_path, model_names=opt.model_names, dataset_names=opt.dataset_names)

    if opt.fig != 'diag':
        print(f'Done! See {out_path} for the saved figure.')
    else:
        print('Done!')


if __name__ == "__main__":
    main()
