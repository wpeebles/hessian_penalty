"""
Create a high-quality disentanglement visualization for network checkpoints. Each row in a saved video corresponds to
interpolating a single z component from -extent to +extent (usually extent=2 but you can control it as an command line
argument). The i-th row controls the i-th z component.

Note that models are automatically visualized at various intervals during training. You can use this
script if you want to visualize a checkpoint later. Also, these visualizations are produced at a much higher quality.
"""

from dnnlib import EasyDict
import dnnlib.tflib as tflib
import argparse
import os
import config
from training.vis_tools import make_h264_mp4_video, sample_interp_zs
from training.misc import load_pkl, find_pkl
import numpy as np
from glob import glob
from download import find_model


def main():
    parser = argparse.ArgumentParser(description='Visualize the Disentanglement of ProgressiveGAN w/ Hessian Penalty')

    # Model/Dataset Parameters:
    parser.add_argument('--models', required=True, nargs='+',
                        help='Either the number of experiment in results directory, or a direct path to a .pkl '
                             'network checkpoint. You can specify multiple experiments/pkls '
                             'to generate visuals for all of them with one call to this script.')
    parser.add_argument('--snapshot_kimgs', default=['latest'], nargs='+',
                        help='network-snapshot-<snapshot_kimg>.pkl to evaluate. This should either be "latest" or '
                             'a list of length equal to exp_numbers (each model needs a snapshot_kimg). '
                             'Not used if you are passing-in direct paths to .pkl checkpoints using --models.')
    parser.add_argument('--seeds', type=int, default=[0], nargs='+',
                        help='Seed for sampling the latent noise')
    parser.add_argument('--nz_per_vid', default=12, type=int,
                        help='Number of z components to visualize per-video. This controls the "height" of the '
                             'generated videos.')
    parser.add_argument('--samples', default=8, type=int,
                        help='Number of z samples to use (=interp_batch_size). This controls the "width" of the '
                             'generated videos.')
    parser.add_argument('--steps', default=90, type=int,
                        help='Number of frames in video (=granularity of interpolation).')
    parser.add_argument('--extent', default=2.0, type=float,
                        help='How "far" to move the z components (from -extent to extent)')
    parser.add_argument('--minibatch_size', default=100, type=int,
                        help='Batch size to use when generating frames. If you get memory errors, try decreasing this.')
    parser.add_argument('--interpolate_pre_norm', action='store_true', default=False,
                        help='If specified, interpolations are performed before the first pixel norm layer in G.'
                             'You should use this when nz is small (e.g., CLEVR-U).')
    parser.add_argument('--no_loop', action='store_true', default=False,
                        help='If specified, saved video will not "loop".')
    parser.add_argument('--stills_only', action='store_true', default=False,
                        help='If specified, only save frames instead of an mp4 video.')
    parser.add_argument('--n_frames_to_save', type=int, default=0,
                        help='Number of "flattened" frames from video to save to png (0=disable).')
    parser.add_argument('--transpose', action='store_true', default=False,
                        help='If specified, flips columns with rows in the video.')
    parser.add_argument('--pad_x', default=0, type=int,
                        help='Padding between samples in video. WARNING: This can '
                             'cause weird problems with the video when '
                             'nz_per_vid > 1, so be careful using this.')
    parser.add_argument('--pad_y', default=0, type=int,
                        help='Padding between rows in video. WARNING: This can '
                             'cause weird problems with the video when '
                             'nz_per_vid > 1, so be careful using this.')

    opt = parser.parse_args()
    opt = EasyDict(vars(opt))
    if opt.pad_x > 0 or opt.pad_y > 0:
        print('Warning: Using non-zero pad_x or pad_y can '
              'cause moviepy to take a long time to make the video. '
              'Also, there might be problems viewing the video with some applications '
              'such as QuickTime.')

    if os.path.isdir(opt.models[0]):
        opt.models = sorted(glob(f'{opt.models[0]}/*/*.pkl'))
    model_paths = [find_model(model) for model in opt.models]
    opt.models = [m1 if m1 else m2 for m1, m2 in zip(model_paths, opt.models)]

    if len(opt.snapshot_kimgs) == 1 and len(opt.models) > 1:
        assert opt.snapshot_kimgs[0] == 'latest'
        opt.snapshot_kimgs = ['latest'] * len(opt.models)
    else:
        assert len(opt.snapshot_kimgs) == len(opt.models)

    for model, snapshot_kimg in zip(opt.models, opt.snapshot_kimgs):
        for seed in opt.seeds:
            opt.exp_number = model
            opt.snapshot_kimg = snapshot_kimg
            opt.seed = seed
            run(opt)


def run(opt):
    # Find and load the network checkpoint:
    if not opt.exp_number.endswith('.pkl'):
        results_dir = os.path.join(os.getcwd(), config.result_dir)
        resume_pkl = find_pkl(results_dir, int(opt.exp_number), opt.snapshot_kimg)
    else:
        resume_pkl = opt.exp_number
    tflib.init_tf()
    _, _, Gs = load_pkl(resume_pkl)
    print(f'Visualizing pkl: {resume_pkl} with seed={opt.seed}')

    # Sample latent noise for making the video frames:
    nz = Gs.input_shapes[0][1]
    np.random.seed(opt.seed)
    z = sample_interp_zs(nz, opt.samples, interp_steps=opt.steps, extent=opt.extent,
                         apply_norm=not opt.interpolate_pre_norm)
    if nz < 12 and not opt.interpolate_pre_norm:
        print(f'This model uses a small z vector (nz={nz}); you might want to add '
              f'--interpolate_pre_norm to your command.')

    # Create directory for saving visualizations (hessian_penalty/visuals/{experiment_name}/seed_{N}):
    checkpoint_kimg = resume_pkl.split('network-snapshot-')[-1].split('.pkl')[0]
    checkpoint_kimg = checkpoint_kimg.split('/')[-1]
    exp_name = resume_pkl.split('/')[-2]
    out_path = os.path.join(config.vis_dir, exp_name, checkpoint_kimg, f'seed_{opt.seed}')
    os.makedirs(out_path, exist_ok=True)

    # Generate and save the video(s):
    make_h264_mp4_video(Gs, z, interp_labels=None, minibatch_size=opt.minibatch_size,
                        interp_steps=opt.steps, interp_batch_size=opt.samples, nz_per_vid=min(nz, opt.nz_per_vid),
                        perfect_loop=not opt.no_loop, vis_path=out_path, use_pixel_norm=opt.interpolate_pre_norm,
                        stills_only=opt.stills_only, transpose=opt.transpose, pad_x=opt.pad_x, pad_y=opt.pad_y,
                        n_frames=opt.n_frames_to_save)
    print(f'Generated visuals can be found at {out_path}')


if __name__ == "__main__":
    main()
