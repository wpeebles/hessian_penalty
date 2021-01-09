"""
Evaluate network checkpoints on FID and/or PPL.

Note that models are automatically evaluated at various intervals during training. You can use this
script if you want to evaluate a checkpoint afterwards.
"""

import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib

import config
from metrics import metric_base
from training import misc

from glob import glob
import argparse
import os
from download import find_dataset, find_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate ProgressiveGAN w/ Hessian Penalty on Various Metrics')

    # Model/Dataset Parameters:
    parser.add_argument('--model', required=True, help='Either the number of experiment in results directory or a path to a .pkl checkpoint.')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs to evaluate with')
    parser.add_argument('--snapshot_kimg', default='latest', help='network-snapshot-<snapshot_kimg>.pkl to evaluate')
    parser.add_argument('--dataset', type=str, default='edges_and_shoes', help='Name of TFRecords directory in datasets/ folder to run metrics with.')
    parser.add_argument('--resolution', type=int, default=128, help='Resolution of real data to evaluate with.')
    parser.add_argument('--metrics', nargs='+', default=['FID', 'PPL'], help='Metrics to run. Must specify at least one')

    opt = parser.parse_args()
    opt = EasyDict(vars(opt))
    assert opt.num_gpus in [1, 2, 4, 8]
    if os.path.isdir(opt.model):  # If you pass a directory organized as DIRECTORY/dataset/model.pkl, it will iterate:git
        paths = sorted(glob(f'{opt.model}/*/*.pkl'))
        for path in paths:
            opt.model = path
            if 'clevr_simple' in path or 'clevr_u' in path:
                opt.dataset = 'clevr_simple'
            elif 'clevr_complex' in path:
                opt.dataset = 'clevr_two_obj'
            elif 'edgeshoes' in path:
                opt.dataset = 'edges_and_shoes'
            elif 'clevr_1fov' in path:
                opt.dataset = 'clevr_1fov'
            else:
                print(f'Couldn\'t find dataset for {path}')
                raise NotImplementedError
            run(opt)
    else:
        run(opt)

# ----------------------------------------------------------------------------


def run_pickle(submit_config, metric_args, network_pkl, dataset_args, mirror_augment, use_RA=True):
    ctx = dnnlib.RunContext(submit_config)
    tflib.init_tf()
    print('Evaluating %s metric on network_pkl "%s"...' % (metric_args.name, network_pkl))
    metric = dnnlib.util.call_func_by_name(**metric_args)
    print()
    metric.run(network_pkl, dataset_args=dataset_args, mirror_augment=mirror_augment, num_gpus=submit_config.num_gpus, use_RA=use_RA)
    print()
    ctx.close()

# ----------------------------------------------------------------------------


def run_snapshot(submit_config, metric_args, run_id, snapshot):
    ctx = dnnlib.RunContext(submit_config)
    tflib.init_tf()
    print('Evaluating %s metric on run_id %s, snapshot %s...' % (metric_args.name, run_id, snapshot))
    run_dir = misc.locate_run_dir(run_id)
    network_pkl = misc.locate_network_pkl(run_dir, snapshot)
    metric = dnnlib.util.call_func_by_name(**metric_args)
    print()
    metric.run(network_pkl, run_dir=run_dir, num_gpus=submit_config.num_gpus)
    print()
    ctx.close()

# ----------------------------------------------------------------------------


def run_all_snapshots(submit_config, metric_args, run_id):
    ctx = dnnlib.RunContext(submit_config)
    tflib.init_tf()
    print('Evaluating %s metric on all snapshots of run_id %s...' % (metric_args.name, run_id))
    run_dir = misc.locate_run_dir(run_id)
    network_pkls = misc.list_network_pkls(run_dir)
    metric = dnnlib.util.call_func_by_name(**metric_args)
    print()
    for idx, network_pkl in enumerate(network_pkls):
        ctx.update('', idx, len(network_pkls))
        metric.run(network_pkl, run_dir=run_dir, num_gpus=submit_config.num_gpus)
    print()
    ctx.close()

# ----------------------------------------------------------------------------


def run(opt):
    submit_config = dnnlib.SubmitConfig()

    # Which metrics to evaluate?
    metrics = []
    if 'FID' in opt.metrics:
        metrics.append(metric_base.fid50k)
    if 'PPL' in opt.metrics:
        metrics.append(metric_base.ppl_zend_v2)

    model_pth = find_model(opt.model)
    if not model_pth:
        # Find the snapshot:
        if opt.model.endswith('.pkl'):
            model_pth = opt.model
        else:
            model_pth = misc.find_pkl(os.path.join(os.getcwd(), config.result_dir), int(opt.model), opt.snapshot_kimg)

    # Define dataset:
    dataset = find_dataset(opt.dataset)
    if not dataset:
        dataset = os.path.join(os.getcwd(), config.data_dir, opt.dataset)

    tasks = []
    tasks += [EasyDict(run_func_name='evaluate.run_pickle', network_pkl=model_pth, use_RA=True,
                       dataset_args=EasyDict(tfrecord_dir=dataset, shuffle_mb=0, resolution=opt.resolution),
                       mirror_augment=False)]

    submit_config.num_gpus = opt.num_gpus

    # Execute.
    submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    submit_config.run_dir_ignore += config.run_dir_ignore
    for task in tasks:
        for metric in metrics:
            submit_config.run_desc = '%s-%s' % (task.run_func_name, metric.name)
            if task.run_func_name.endswith('run_snapshot'):
                submit_config.run_desc += '-%s-%s' % (task.run_id, task.snapshot)
            if task.run_func_name.endswith('run_all_snapshots'):
                submit_config.run_desc += '-%s' % task.run_id
            submit_config.run_desc += '-%dgpu' % submit_config.num_gpus
            dnnlib.submit_run(submit_config, metric_args=metric, **task)
    print('Done with %s.' % model_pth)
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
