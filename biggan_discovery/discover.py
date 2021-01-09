"""
Learns a matrix of Z-Space directions using a pre-trained BigGAN Generator.
Modified from train.py in the PyTorch BigGAN repo.
"""

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim

import utils
import train_fns
from sync_batchnorm import patch_replication_callback

from torch.utils.tensorboard import SummaryWriter
from hessian_penalty import hessian_penalty
from direction_utils import visualize_directions, load_G, get_direction_padding_fn, init_wandb, download_G
from layers import fast_gram_schmidt


class DataParallelLoss(nn.Module):
    """
    This is simply a wrapper class to compute the Hessian Penalty efficiently over several GPUs
    """

    def __init__(self, G):
        super(DataParallelLoss, self).__init__()
        self.G = G

    def forward(self, z, y, w, Q):
        penalty = hessian_penalty(self.G, z, c=y, w=w, G_z=None, Q=Q, multiple_layers=False)
        return penalty


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):
    if config['wandb_entity'] is not None:
        init_wandb(config, config['experiment_name'], config['wandb_entity'], 'imagenet')
    if config["G_path"] is None:  # Download a pre-trained G if necessary
        download_G()
        config["G_path"] = 'checkpoints/138k'
    G, state_dict, device, experiment_name = load_G(config)
    # If parallel, parallelize the GD module
    if config['parallel']:
        G = nn.DataParallel(DataParallelLoss(G))
        if config['cross_replica']:
            patch_replication_callback(G)

    num_gpus = torch.cuda.device_count()
    print(f'Using {num_gpus} GPUs')

    # If search_space != 'all', then we need to pad the z components that we are leaving alone:
    pad = get_direction_padding_fn(config)
    direction_size = config['dim_z'] if config['search_space'] == 'all' else config['ndirs']
    # A is our (ndirs, |z|) matrix of directions, where ndirs indicates the number of directions we want to learn
    if config['load_A'] == 'coords':
        print('Initializing with standard basis directions')
        A = torch.nn.Parameter(torch.eye(config['ndirs'], direction_size, device=device), requires_grad=True)
    elif config['load_A'] == 'random':
        print('Initializing with random directions')
        A = torch.nn.Parameter(torch.empty(config['ndirs'], direction_size, device=device), requires_grad=True)
        torch.nn.init.kaiming_normal_(A)
    else:
        raise NotImplementedError
    # We only learn A; G is left frozen during training:
    optim = torch.optim.Adam(params=[A], lr=config['A_lr'])

    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.module.G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])

    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.module.G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    interp_z, interp_y = utils.prepare_z_y(config["n_samples"], G.module.G.dim_z,
                                           config['n_classes'], device=device,
                                           fp16=config['G_fp16'])
    interp_z.sample_()
    interp_y.sample_()

    if config['fix_class'] is not None:
        y_ = y_.new_full(y_.size(), config['fix_class'])
        fixed_y = fixed_y.new_full(fixed_y.size(), config['fix_class'])
        interp_y = interp_y.new_full(interp_y.size(), config['fix_class'])

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    iters_per_epoch = 1000
    dummy_loader = [None] * iters_per_epoch  # We don't need any real data

    path_size = config['path_size']
    # Simply stores a |z|-dimensional one-hot vector indicating each direction we are learning:
    direction_indicators = torch.eye(config['ndirs']).to(device)

    G.eval()

    G.module.optim = optim

    writer = SummaryWriter('%s/%s' % (config['logs_root'], experiment_name))
    sample_sheet = train_fns.save_and_sample(G.module.G, None, G.module.G, z_, y_, fixed_z, fixed_y,
                                             state_dict, config, experiment_name)
    writer.add_image('samples', sample_sheet, 0)

    interp_y_ = G.module.G.shared(interp_y)
    # Make directions orthogonal via Gram Schmidt:
    Q = pad(fast_gram_schmidt(A)) if not config["no_ortho"] else pad(A)

    if config["vis_during_training"]:
        print("Generating initial visualizations...")
        interp_vis = visualize_directions(G.module.G, interp_z, interp_y_, path_sizes=path_size, Q=Q,
                                          high_quality=False, npv=1)
        for w_ix in range(config['ndirs']):
            writer.add_video('G_ema/w%03d' % w_ix, interp_vis[w_ix], 0, fps=24)

    for epoch in range(state_dict['epoch'], config['num_epochs']):
        if config['pbar'] == 'mine':
            pbar = utils.progress(dummy_loader, displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(dummy_loader)
        for i, _ in enumerate(pbar):
            state_dict['itr'] += 1
            z_.sample_()
            if config['fix_class'] is None:
                y_.sample_()
            y = G.module.G.shared(y_)
            sampled_directions = torch.randint(low=0, high=config['ndirs'], size=(G_batch_size,), device=device)
            # Distances are sampled from U[-path_size, path_size]:
            distances = torch.rand(G_batch_size, 1, device=device).mul(2 * path_size).add(-path_size)
            # w_sampled is an (N, ndirs)-shaped tensor. If i indexes batch elements and j indexes directions, then
            # w_sampled[i, j] represents how far we will move z[i] in the direction Q[j]. The final z[i] will be the sum
            # over all directions stored in the rows of Q.
            w_sampled = direction_indicators[sampled_directions] * distances
            # TODO: The Q.repeat below is a DataParallel hack to make sure each GPU gets the same copy of the Q matrix.
            # There is almost certainly a cleaner way to do this.
            # Hessian Penalty taken w.r.t. w_sampled, NOT z:
            penalty = G(z_, y, w=w_sampled, Q=Q.repeat(num_gpus, 1)).mean()

            optim.zero_grad()
            penalty.backward()
            optim.step()
            # re-orthogonalize A for visualizations and the next training iteration:
            Q = pad(fast_gram_schmidt(A)) if not config["no_ortho"] else pad(A)

            # Log metrics to TensorBoard/WandB:
            cur_training_iter = epoch * iters_per_epoch + i
            writer.add_scalar('Metrics/hessian_penalty', penalty.item(), cur_training_iter)
            writer.add_scalar('Metrics/direction_norm', A.pow(2).mean().pow(0.5).item(), cur_training_iter)

            # Save directions and log visuals:
            if not (state_dict['itr'] % config['save_every']):
                torch.save(A.cpu().detach(), '%s/%s/A_%06d.pt' %
                           (config['weights_root'], experiment_name, cur_training_iter))
                if config["vis_during_training"]:
                    interp_vis = visualize_directions(G.module.G, interp_z, interp_y_, path_sizes=path_size, Q=Q,
                                                      high_quality=False, npv=1)
                    for w_ix in range(config['ndirs']):
                        writer.add_video('G_ema/w%03d' % w_ix, interp_vis[w_ix], cur_training_iter, fps=24)

        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
