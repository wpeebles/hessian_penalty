import torch
import torchvision
import numpy as np
import imageio
import moviepy.editor
from there_and_back import mid_right_mid_left_mid, left_to_right
import utils
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
import os
import shutil


def save_interp_video(images, interp_steps, path=None, norm=False, nrow=-1):
    """
    images: interp_steps * batch_size x C x H x W tensor (rank 4)
    """
    N = images.size(0)
    bs = N // interp_steps
    assert N == bs * interp_steps
    frames = []
    if nrow == -1:
        nrow = bs
    for i in range(interp_steps):
        ixs = list(range(i, N, interp_steps))
        frame_i = torchvision.utils.make_grid(images[ixs], nrow=nrow, pad_value=-1)
        frames.append(frame_i)
    pyt_frames = torch.stack(frames)
    if path:
        frames = (pyt_frames.cpu().numpy() + 1) / 2
        frames = 255 * frames
        frames = np.rint(frames).clip(0, 255).astype(np.uint8)
        frames = np.transpose(frames, (0, 2, 3, 1))
        imageio.mimsave(path, frames, duration=1/24)
    if norm:
        pyt_frames = (pyt_frames.unsqueeze(0).cpu().numpy() + 1) / 2
    return pyt_frames


def split_interp_videos(images, interp_steps, batch_size, base_path=None):
    nz = images.shape[0] // (interp_steps * batch_size)
    assert nz * interp_steps * batch_size == images.shape[0]
    offset = interp_steps * batch_size
    if base_path:
        base_path += '/z%03d.gif'
    return [save_interp_video(images[z*offset:(z+1)*offset], interp_steps, norm=True,
                              path=(base_path % z if base_path else None)) for z in range(nz)]


@torch.no_grad()
def high_fps_h264(images, interp_steps, batch_size, path, fps=60, loop=True):
    duration = (1 + loop) * interp_steps / fps
    all_images = images
    offset = interp_steps * batch_size
    nz = all_images.size(0) // offset
    assert nz * offset == all_images.size(0)
    videos = []
    for i in range(nz):
        # interp_steps x C x H x W
        video_i = save_interp_video(all_images[i * offset: (i + 1) * offset], interp_steps, nrow=batch_size)
        videos.append(video_i)
    frames = []
    for t in range(interp_steps):
        frame = []
        for i in range(nz):
            frame.append(videos[i][t])
        frame = torch.stack(frame)  # nz x C x Hgrid x Wgrid (rank 4)
        frame = torchvision.utils.make_grid(frame, nrow=1, padding=0, pad_value=-1)  # rank 3
        frames.append(frame)
    pyt_frames = torch.stack(frames)
    frames = (pyt_frames.cpu().numpy() + 1) / 2
    frames = 255 * frames
    frames = np.rint(frames).clip(0, 255).astype(np.uint8)
    frames = np.transpose(frames, (0, 2, 3, 1))
    if loop:
        frames = np.concatenate([frames, frames[::-1]], axis=0)
    frames = [frames[i] for i in range(frames.shape[0])] + [frames[-1]]

    def make_frame(t):
        return frames.pop()

    moviepy.editor.VideoClip(make_frame, duration=duration).write_videofile(path, fps=fps, codec='libx264', bitrate='50M')


@torch.no_grad()
def batch_directions(z, y, Q, path_sizes, step_vals, subtract_projection=True):
    """
    This function takes an input batch of z vectors (and corresponding class label vectors y)
    and applies the directions in Q to the batch of z vectors.

    :param z: (N, nz) tensor of base random noise to manipulate with directions
    :param y: (N, D) tensor of class vectors
    :param Q: (ndirs, nz) matrix of z-space directions
    :param path_sizes: (ndirs,) tensor indicating how far to travel in each direction
    :param step_vals: (interp_steps,) tensor controlling the granularity of the interpolation
    :param subtract_projection: bool, whether or not to "remove" each direction from the sampled z vectors
    :return: z: (N * ndirs * interp_steps, nz) tensor, y: (N * ndirs * interp_steps) tensor
             containing all z's and y's needed to create the visualizations
    """
    interp_steps = step_vals.size(0)
    N, nz = z.size()
    ndirs = Q.size(0)
    z = z.view(1, N, 1, nz).repeat(ndirs, 1, interp_steps, 1)  # .view(N * ndirs * interp_steps, nz)
    if subtract_projection:
        # The projection will be the same across the interp_steps dimension, so we can just pick-out the first step:
        z_proj = z[:, :, 0, :].view(ndirs * N, nz)
        Q_proj = Q.repeat_interleave(N, dim=0)
        projection = (z_proj * Q_proj).sum(dim=1, keepdims=True) / Q_proj.pow(2).sum(dim=1, keepdims=True) * Q_proj
        z -= projection.view(ndirs, N, 1, nz)
    path_sizes = path_sizes.view(ndirs, 1, 1, 1)
    step_vals = step_vals.view(1, 1, interp_steps, 1)
    Q = Q.view(ndirs, 1, 1, nz)
    z += step_vals * path_sizes * Q
    z = z.view(N * ndirs * interp_steps, nz)
    y = y.repeat_interleave(interp_steps, dim=0).repeat(ndirs, 1)
    return z, y


@torch.no_grad()
def visualize_directions(G, z, y, path_sizes=5, interp_steps=24,
                         Q=None, base_path=None, step_override=None, interp_mode='smooth_center',
                         directions_to_vis=None, high_quality=True, npv=8, fps=60, minibatch_size=80, quiet=True):
    if directions_to_vis is None:
        directions_to_vis = list(range(Q.size(0)))
    else:
        Q = Q[directions_to_vis]
    num_directions = Q.size(0)
    batch_size = z.size(0)
    if isinstance(path_sizes, int) or isinstance(path_sizes, float):
        path_sizes = path_sizes * torch.ones(num_directions)
    loop = False  # smooth and smooth_center already loop by default
    if interp_mode == 'smooth_center':
        step_vals = mid_right_mid_left_mid(interp_steps // 2, False)
    elif interp_mode == 'smooth':
        step_vals = left_to_right(interp_steps, False)
    elif interp_mode == 'linear':
        step_vals = torch.linspace(-1.0, 1.0, interp_steps)
        loop = True
    else:
        raise NotImplementedError
    z, y = batch_directions(z, y, Q, path_sizes.cuda(), step_vals.cuda())
    out = []
    itr = range(0, z.size(0), minibatch_size)
    if not quiet:
        itr = tqdm(itr)
    for batch_ix in itr:
        out.append(G(z[batch_ix:batch_ix + minibatch_size], y[batch_ix:batch_ix+minibatch_size]).cpu())
    out = torch.cat(out, 0)
    if not high_quality:  # Return a list of videos that can be logged to WandB/TensorBoard:
        return split_interp_videos(out, interp_steps, batch_size, base_path=base_path)
    else:  # Directly save high quality mp4 videos of the directions:
        for z_i, ix in enumerate(range(0, out.size(0), npv * batch_size * interp_steps)):
            path_name = '_'.join(['Q%03d' % s for s in directions_to_vis[npv * z_i: npv * (z_i + 1)]])
            frames = out[ix: ix + npv * batch_size * interp_steps]
            num_frames = interp_steps
            if step_override is not None:
                num_frames = len(step_override[z_i])
                frames = frames.reshape((npv, batch_size, interp_steps, *frames.size()[1:]))
                frames = frames[:, :, step_override[z_i]]
                frames = frames.reshape((npv * batch_size * num_frames, *frames.size()[3:]))
            high_fps_h264(frames, num_frames, batch_size, '%s/%03d_%s.mp4' % (base_path, z_i, path_name),
                          loop=loop, fps=fps)


def get_direction_padding_fn(config):
    # Determine which z components will be manipulated:
    if config["search_space"] == "all":  # Manipulate every z component:
        start_z = 0
        end_z = config["dim_z"]
    else:  # Manipulate just one segment of z components:
        assert config["dim_z"] == 120
        config["ndirs"] = 40
        if config["search_space"] == "coarse":
            start_z = 0
            end_z = 40
        elif config["search_space"] == "mid":
            start_z = 40
            end_z = 80
        elif config["search_space"] == "fine":
            start_z = 80
            end_z = 120
    # zero-pad directions to ensure z components that we do not manipulate
    # will be left constant when the direction is added:
    pad = lambda x: F.pad(x, (start_z, config["dim_z"] - end_z, 0, 0))
    return pad


def load_G(config):
    """
    Loads a pre-trained BigGAN generator network (exponential moving average variant).
    """
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]

    config = utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))

    G = model.Generator(**{**config, 'skip_init': True, 'no_optim': True}).to(device)

    # FP16? (Note: half/mixed-precision is untested with the direction discovery code)
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()

    print(G)
    print('Number of params in G: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # Load the pre-trained G_ema model as "G"
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(None, None, state_dict,
                           None, None,
                           config['load_weights'] if config['load_weights'] else None,
                           G, load_optim=False, strict=False, direct_path=config['G_path'])

    G.to(device)
    # Override G's optimizer to only optimize the direction matrix A:
    for param in G.parameters():
        param.requires_grad = False
    G.optim = None
    G.eval()
    return G, state_dict, device, experiment_name


def download_G(root='checkpoints'):
    """Downloads a 128x128 BigGAN checkpoint to use for direction discovery."""
    # This is the corresponding file ID for the PyTorch BigGAN 138k checkpoint available at the following URL:
    # https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view
    ID = '1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW'
    path = f'{root}/138k'
    if not os.path.isdir(path):
        zip_path = f'{path}.zip'
        print(f'Downloading BigGAN checkpoint directory to {path}')
        download_file_from_google_drive('1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW', root)
        shutil.move(f'{root}/{ID}', zip_path)
        extract_archive(zip_path, remove_finished=True)
    else:
        print(f'Resuming from checkpoint at {path}.')


def init_wandb(opt, name, entity, group, project='hessian_penalty', sync_tensorboard=True):
    """If using WandB, starts project tracking. Code from Tim Brooks."""
    import wandb
    import pathlib

    wandb_runs = wandb.Api().runs(path=f'{entity}/{project}', order='-created_at')

    if not wandb_runs:
        wandb_index = 0
    else:
        wandb_prev_name = wandb_runs[0].name
        wandb_index = int(wandb_prev_name.split('_')[0]) + 1

    wandb_name = f'{wandb_index:04d}_{name}'
    log_dir = str(pathlib.Path(__file__).parent)
    wandb.init(config=opt, name=wandb_name, dir=log_dir, entity=entity,
               project=project, group=group, sync_tensorboard=sync_tensorboard)
