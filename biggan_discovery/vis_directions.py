"""
Modified from the train.py in the PyTorch BigGAN repo.
"""

import os
import torch
import torch.nn as nn
import torch.optim

import utils
from sync_batchnorm import patch_replication_callback

from direction_utils import visualize_directions, load_G, get_direction_padding_fn, download_G
from layers import fast_gram_schmidt


def run(config):
    if config["G_path"] is None:  # Download a pre-trained G if necessary
        download_G()
        config["G_path"] = f'checkpoints/138k'
    G, state_dict, device, experiment_name = load_G(config)
    # If parallel, parallelize the GD module
    if config['parallel']:
        G = nn.DataParallel(G)
        if config['cross_replica']:
            patch_replication_callback(G)
    pad = get_direction_padding_fn(config)
    ndirs = config["ndirs"] if config["directions_to_vis"] is None else len(config["directions_to_vis"])

    path_sizes = torch.tensor([config["path_size"]] * ndirs, dtype=torch.float32)

    interp_z, interp_y = utils.prepare_z_y(config["n_samples"], G.module.dim_z,
                                           config['n_classes'], device=device,
                                           fp16=config['G_fp16'])
    interp_z.sample_()
    interp_y.sample_()

    if config['fix_class'] is not None:
        interp_y = interp_y.new_full(interp_y.size(), config['fix_class'])

    interp_y_ = G.module.shared(interp_y)

    direction_size = config["dim_z"] if config["search_space"] == "all" else config["ndirs"]
    if config['load_A'] == 'random':
        print('Visualizing RANDOM directions')
        A = torch.randn(ndirs, direction_size)
        A_name = 'random'
        nn.init.kaiming_normal_(A)
    elif config['load_A'] == 'coord':
        print('Visualizing COORDINATE directions')
        A = torch.eye(ndirs, direction_size)
        A_name = 'coord'
    else:
        print('Visualizing PRE-TRAINED directions')
        A = torch.load(config["load_A"])
        A_name = 'pretrained'

    A = A.cuda()
    Q = pad(fast_gram_schmidt(A)) if not config["no_ortho"] else pad(A)

    visuals_dir = f'visuals/{experiment_name}/{A_name}'
    os.makedirs(visuals_dir, exist_ok=True)
    print('Generating interpolation videos...')
    visualize_directions(G, interp_z, interp_y_, path_sizes=path_sizes, Q=Q, base_path=visuals_dir, interp_steps=180,
                         interp_mode='smooth_center', high_quality=True, quiet=False,
                         minibatch_size=config["val_minibatch_size"], directions_to_vis=config["directions_to_vis"])


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
