"""
Start a training job for ProgressiveGAN with the Hessian Penalty.

For the training hyper-parameters of the Edges+Shoes/CLEVR models in our paper, check out the scripts folder.
"""

import copy
import dnnlib
from dnnlib import EasyDict
import config
from metrics import metric_base
import argparse
import os
from training.misc import find_pkl
from download import find_dataset


def main():
    parser = argparse.ArgumentParser(description='Train ProgressiveGAN w/ Hessian Penalty')

    # Model/Dataset Parameters:
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs to train with')
    parser.add_argument('--dataset', type=str, default='edges_and_shoes', help='Name of TFRecords directory in datasets/ folder to train on.')
    parser.add_argument('--resolution', type=int, default=128, help='Maximum resolution (after progressive growth) to train at.')
    parser.add_argument('--nz', type=int, default=12, help='Number of components in G\'s latent space.')
    parser.add_argument('--total_kimg', type=int, default=30000, help='How long to train for')
    parser.add_argument('--resume_exp', type=int, default=None, help='If specified, resumes training from a snapshot in the results directory with job number resume_exp')
    parser.add_argument('--resume_snapshot', default='latest', help='If using resume_exp, you can override this default to resume training from a specific checkpoint')
    parser.add_argument('--seed', type=int, default=1000, help='NumPy Random Seed')

    # Hessian Penalty Parameters:
    parser.add_argument('--hp_lambda', type=float, default=0.1,
                        help='Loss weighting of the Hessian Penalty regularization term. '
                             'When fine-tuning with the Hessian Penalty, this value '
                             'is the maximum weighting once the loss term is fully ramped-up.'
                             'Set to 0.0 to disable the Hessian Penalty.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='The granularity of the finite differences approximation. '
                             'When changing this value from 0.1, you will likely need to change '
                             'hp_lambda as well to get optimal results.')
    parser.add_argument('--num_rademacher_samples', type=int, default=2,
                        help='The number of Rademacher vectors to be sampled per-batch element '
                             'when estimating the Hessian Penalty. Must be >=2. Setting this '
                             'parameter larger can result in GPU out-of-memory for smaller GPUs!')
    parser.add_argument('--layers_to_reg', nargs='*', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='Which activations to directly apply the Hessian Penalty to? For 128x128 '
                             'resolution, you can choose any layers 1-13. '
                             '(1=first fully-connected layer, 2-12=convolutional layers, 13=RGB activations)')
    parser.add_argument('--warmup_kimg', type=int, default=10000,
                        help='Over how many kimg (=thousands of real images fed to D) should '
                             'the Hessian Penalty weight be linearly ramped-up? Setting this '
                             'value too low can sometimes result in training instability when fine-tuning.'
                             'The warmup will start at hp_start_kimg.')
    parser.add_argument('--hp_start_kimg', type=int, default=0,
                        help='To train from scratch with the Hessian Penalty, set this to 0. '
                             'If >0, the Hessian Penalty will be phased-in starting at the '
                             'specified iteration. Alternatively, you can load '
                             'a checkpoint and simply set this to 0 to fine-tune a pretrained model.'
                             'Note that we haven\'t tested setting this > 0.')

    # InfoGAN Parameters:
    parser.add_argument('--infogan_lambda', type=float, default=0.0,
                        help='Loss weighting for InfoGAN loss on latent vectors. Set to 0.0 to disable InfoGAN loss.')
    parser.add_argument('--infogan_nz', type=int, default=0,
                        help='Number of Z components to regularize with InfoGAN\'s mutual information loss. '
                             'For a (batch_size, nz) Z vector, you can access the regularized components with '
                             'Z[:, :infogan_nz].')

    # Visualization and Metric Parameters:
    parser.add_argument('--compute_interp_ticks', type=int, default=5,
                        help='Training ticks between computing disentanglement visualizations (interpolations).')
    parser.add_argument('--compute_metrics_ticks', type=int, default=5,
                        help='Training ticks between computing FID (and optionally PPL; see next parameter).')
    parser.add_argument('--metrics', nargs='*', default=['FID', 'PPL'],
                        help='Which metrics to compute during training.')
    parser.add_argument('--dashboard_api', type=str, choices=['tensorboard', 'wandb'], default='tensorboard',
                        help='Which training dashboard software to use for logging visualizations/metrics/losses/etc; '
                             'either TensorBoard (default) or WandB. If you choose to use WandB, you will first need '
                             'an account on wandb.com.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='If using WandB, the entity for logging'
                                                                       '(e.g., your username)')

    opt = parser.parse_args()
    opt = EasyDict(vars(opt))
    verify_opt(opt)
    run(opt)


def build_job_name(opt):
    """Returns a job title for the results directory."""
    if opt.hp_lambda > 0:
        job_name = f'hessian_penalty_{opt.dataset}_nz-{opt.nz}_w-{opt.hp_lambda}'
    elif opt.infogan_lambda > 0:
        job_name = f'infogan_{opt.dataset}_nz-{opt.nz}_inz-{opt.infogan_nz}'
    else:
        job_name = f'progan_{opt.dataset}_nz-{opt.nz}'
    return job_name


def run(opt):
    """Sets-up all of the parameters necessary to start a ProgressiveGAN training job."""
    desc = build_job_name(opt)                                              # Description string included in result subdir name.
    train = EasyDict(run_func_name='training.training_loop.training_loop')  # Options for training loop.
    G = EasyDict(func_name='training.networks_progan.G_paper')              # Options for generator network.
    D = EasyDict(func_name='training.networks_progan.D_paper')              # Options for discriminator network.
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                   # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                   # Options for discriminator optimizer.
    G_loss = EasyDict(func_name='training.loss.G_wgan')                     # Options for generator loss.
    D_loss = EasyDict(func_name='training.loss.D_wgan_gp')                  # Options for discriminator loss.
    sched = EasyDict()                                                      # Options for TrainingSchedule.
    grid = EasyDict(size='1080p', layout='random')                          # Options for setup_snapshot_image_grid().
    submit_config = dnnlib.SubmitConfig()                                   # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': opt.seed}                            # Options for tflib.init_tf().
    metrics = []                                                            # Metrics to run during training.

    if 'FID' in opt.metrics:
        metrics.append(metric_base.fid50k)
    if 'PPL' in opt.metrics:
        metrics.append(metric_base.ppl_zend_v2)
    train.network_metric_ticks = opt.compute_metrics_ticks
    train.interp_snapshot_ticks = opt.compute_interp_ticks

    find_dataset(opt.dataset)

    # Optionally resume from checkpoint:
    if opt.resume_exp is not None:
        results_dir = os.path.join(os.getcwd(), config.result_dir)
        _resume_pkl = find_pkl(results_dir, opt.resume_exp, opt.resume_snapshot)
        train.resume_run_id = opt.resume_exp
        train.resume_snapshot = _resume_pkl
        train.resume_kimg = int(_resume_pkl.split('.pkl')[0][-6:])
        if f'hessian_penalty_{opt.dataset}' not in _resume_pkl and opt.hp_lambda > 0:
            print('When fine-tuning a job that was originally trained without the Hessian Penalty, '
                  'hp_start_kimg is relative to the kimg of the checkpoint being resumed from. '
                  'Hessian Penalty will be phased-in starting at absolute '
                  f'kimg={opt.hp_start_kimg + train.resume_kimg}.')
            opt.hp_start_kimg += train.resume_kimg

    # Set up dataset hyper-parameters:
    dataset = EasyDict(tfrecord_dir=os.path.join(os.getcwd(), config.data_dir, opt.dataset), resolution=opt.resolution)
    train.mirror_augment = False

    # Set up network hyper-parameters:
    G.latent_size = opt.nz
    D.infogan_nz = opt.infogan_nz
    G.infogan_lambda = opt.infogan_lambda
    D.infogan_lambda = opt.infogan_lambda

    # When computing the multi-layer Hessian Penalty, we retrieve intermediate activations by accessing the
    # corresponding tensor's name. Below are the names of various activations in G that we can retrieve:
    activation_type = 'norm'
    progan_generator_layer_index_to_name = {
        1: f'4x4/Dense/Post_{activation_type}',
        2: f'4x4/Conv/Post_{activation_type}',
        3: f'8x8/Conv0_up/Post_{activation_type}',
        4: f'8x8/Conv1/Post_{activation_type}',
        5: f'16x16/Conv0_up/Post_{activation_type}',
        6: f'16x16/Conv1/Post_{activation_type}',
        7: f'32x32/Conv0_up/Post_{activation_type}',
        8: f'32x32/Conv1/Post_{activation_type}',
        9: f'64x64/Conv0_up/Post_{activation_type}',
        10: f'64x64/Conv1/Post_{activation_type}',
        11: f'128x128/Conv0_up/Post_{activation_type}',
        12: f'128x128/Conv1/Post_{activation_type}',
        13: 'images_out'  # final full-resolution RGB activation
    }

    # Convert from layer indices to layer names (which we'll need to compute the Hessian Penalty):
    layers_to_reg = [progan_generator_layer_index_to_name[layer_ix] for layer_ix in sorted(opt.layers_to_reg)]

    # Store the Hessian Penalty parameters in their own dictionary:
    HP = EasyDict(hp_lambda=opt.hp_lambda, epsilon=opt.epsilon, num_rademacher_samples=opt.num_rademacher_samples,
                  layers_to_reg=layers_to_reg, warmup_nimg=opt.warmup_kimg * 1000,
                  hp_start_nimg=opt.hp_start_kimg * 1000)

    # How long to train for (as measured by thousands of real images processed, not gradient steps):
    train.total_kimg = opt.total_kimg

    # We ran the original experiments using 4 GPUs per job. If using a different number,
    # we try to scale batch sizes up or down accordingly in the for-loop below. Note that
    # using other batch sizes is somewhat untested, though!
    submit_config.num_gpus = opt.num_gpus
    sched.minibatch_base = 32
    sched.minibatch_dict = {4: 2048, 8: 1024, 16: 512, 32: 256, 64: 128, 128: 96, 256: 32, 512: 16}
    for res, batch_size in sched.minibatch_dict.items():
        sched.minibatch_dict[res] = int(batch_size * opt.num_gpus / 4)

    # Set-up WandB if optionally using it instead of TensorBoard:
    if opt.dashboard_api == 'wandb':
        init_wandb(opt=opt, name=desc, group=opt.dataset, entity=opt.wandb_entity)

    # Start the training job:
    kwargs = EasyDict(train)
    kwargs.update(HP_args=HP, G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


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


def verify_opt(opt):
    """Sanity check a few parts of the opt (non-exhaustive)."""
    assert opt.nz >= opt.infogan_nz
    assert opt.infogan_nz == 0 or opt.infogan_lambda > 0
    assert len(opt.layers_to_reg) == 0 or min(opt.layers_to_reg) >= 1
    assert opt.epsilon > 0 or opt.hp_lambda == 0
    assert opt.num_rademacher_samples >= 2 or opt.hp_lambda == 0
    assert opt.num_gpus in [1, 2, 4, 8]


if __name__ == "__main__":
    main()
