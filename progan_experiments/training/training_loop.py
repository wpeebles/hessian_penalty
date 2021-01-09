"""Main training script."""

from training.hessian_penalties import get_current_hessian_penalty_loss_weight
from copy import deepcopy
from metrics import metric_base
from training import vis_tools
from training import misc
from training import dataset
import train
import config
from dnnlib.tflib.autosummary import autosummary, build_image_summary
import dnnlib.tflib as tflib
import dnnlib
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress annoying TensorFlow deprecation warnings


# ----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'):  # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2] // 2, 2, s[3] // 2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'):  # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

# ----------------------------------------------------------------------------
# Evaluate time-varying training parameters.


def training_schedule(
        cur_nimg,
        training_set,
        num_gpus,
        lod_initial_resolution=4,        # Image resolution used at the beginning.
        lod_training_kimg=600,      # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg=600,      # Thousands of real images to show when fading in new layers.
        minibatch_base=16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict={},       # Resolution-specific overrides.
        max_minibatch_per_gpu={},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base=0.001,    # Learning rate for the generator.
        G_lrate_dict={},       # Resolution-specific overrides.
        D_lrate_base=0.001,    # Learning rate for the discriminator.
        D_lrate_dict={},       # Resolution-specific overrides.
        lrate_rampup_kimg=0,        # Duration of learning rate ramp-up.
        tick_kimg_base=160,      # Default interval of progress snapshots.
        tick_kimg_dict={4: 160, 8: 140, 16: 120, 32: 100, 64: 80, 128: 60, 256: 40, 512: 30, 1024: 20}):  # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    s.lod = training_set.resolution_log2
    s.lod -= np.floor(np.log2(lod_initial_resolution))
    s.lod -= phase_idx
    if lod_transition_kimg > 0:
        s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch = minibatch_dict.get(s.resolution, minibatch_base)
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

# ----------------------------------------------------------------------------
# Main training script.


def training_loop(
        submit_config,
        HP_args={},       # Options for the Hessian Penalty.
        G_args={},       # Options for generator network.
        D_args={},       # Options for discriminator network.
        G_opt_args={},       # Options for generator optimizer.
        D_opt_args={},       # Options for discriminator optimizer.
        G_loss_args={},       # Options for generator loss.
        D_loss_args={},       # Options for discriminator loss.
        dataset_args={},       # Options for dataset.load_dataset().
        sched_args={},       # Options for train.TrainingSchedule.
        grid_args={},       # Options for train.setup_snapshot_image_grid().
        metric_arg_list=[],       # Options for MetricGroup.
        tf_config={},       # Options for tflib.init_tf().
        G_smoothing_kimg=10.0,     # Half-life of the running average of generator weights.
        D_repeats=1,        # How many times the discriminator is trained per G iteration.
        minibatch_repeats=4,        # Number of minibatches to run before adjusting training parameters.
        reset_opt_for_new_lod=True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
        total_kimg=15000,    # Total length of the training, measured in thousands of real images.
        mirror_augment=False,    # Enable mirror augment?
        drange_net=[-1, 1],   # Dynamic range used when feeding image data to the networks.
        image_snapshot_ticks=1,        # How often to export image snapshots?
        interp_snapshot_ticks=20,       # How often to generate interpolation visualizations in TensorBoard?
        network_snapshot_ticks=5,        # How often to export network snapshots?
        network_metric_ticks=5,        # How often to evaluate network snapshots on specified metrics?
        save_tf_graph=False,    # Include full TensorFlow computation graph in the tfevents file?
        save_weight_histograms=False,    # Include weight histograms in the tfevents file?
        resume_run_id=None,     # Run ID or network pkl to resume training from, None = start from scratch.
        resume_snapshot=None,     # Snapshot index to resume training from, None = autodetect.
        resume_kimg=0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
        resume_time=0.0):     # Assumed wallclock time at the beginning. Affects reporting.

    # Initialize dnnlib and TensorFlow.
    ctx = dnnlib.RunContext(submit_config, train)
    tflib.init_tf(tf_config)

    # Load training set.
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)

    # Create a copy of dataset_args for running the metrics:
    metrics_dataset_args = deepcopy(dataset_args)
    metrics_dataset_args.shuffle_mb = 0

    print('Saving interp videos every %s ticks' % interp_snapshot_ticks)
    print('Saving network snapshot every %s ticks' % network_snapshot_ticks)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
    # G.print_layers(); D.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        Gs_beta = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    # The loss weighting of the Hessian Penalty can be dynamic over training, so we need to use a placeholder:
    hessian_weight = tf.placeholder(tf.float32, name='hessian_weight', shape=[])

    G_opt = tflib.Optimizer(name='TrainG', learning_rate=lrate_in, **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', learning_rate=lrate_in, **D_opt_args)
    reg_ops = []  # Returning the values of the Hessian Penalty/ InfoGAN losses so they can be registered in TensorBoard
    for gpu in range(submit_config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            reals, labels = training_set.get_minibatch_tf()
            reals = process_reals(reals, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss, G_hessian_penalty = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt,
                                                                          training_set=training_set,
                                                                          minibatch_size=minibatch_split,
                                                                          hp_lambda=hessian_weight,
                                                                          HP_args=HP_args,
                                                                          gpu_ix=gpu,
                                                                          lod_in=lod_in,
                                                                          max_lod=training_set.resolution_log2,
                                                                          **G_loss_args)
                if HP_args.hp_lambda > 0:
                    reg_ops += [G_hessian_penalty]
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss, mutual_info = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt,
                                                                    training_set=training_set,
                                                                    minibatch_size=minibatch_split, reals=reals,
                                                                    labels=labels, gpu_ix=gpu,
                                                                    infogan_nz=D_args.infogan_nz,
                                                                    **D_loss_args)
                # print([name for name in D_gpu.trainables.keys()])
                # gps = [weight for name, weight in G_gpu.trainables.items()][0]
                # dps = [weight for name, weight in D_gpu.trainables.items() if 'Q_Encoder' in name][0]
                # gg = autosummary('Loss/G_info_grad', tf.reduce_sum(tf.gradients(mutual_info, gps)[0]**2))
                # dg = autosummary('Loss/D_info_grad', tf.reduce_sum(tf.gradients(mutual_info, dps)[0]**2))
                # reg_ops.extend([dg, gg, dps, gps])
                if G_args.infogan_lambda > 0 or D_args.infogan_lambda > 0:
                    reg_ops += [mutual_info]
            # Note, even though we are adding mutual_info loss here, the only time the loss is non-zero
            # is when infogan_lambda > 0 (in Hessian Penalty experiments, we always set it to zero):
            G_opt.register_gradients(G_loss + G_args.infogan_lambda * mutual_info, G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss) + D_args.infogan_lambda * mutual_info, D_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)

    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels, grid_latents = misc.setup_snapshot_image_grid(G, training_set, **grid_args)
    sched = training_schedule(cur_nimg=total_kimg * 1000, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
    grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch // submit_config.num_gpus)

    print('Setting up snapshot interpolation...')
    nz = G.input_shapes[0][1]
    interp_steps = 24  # Number of frames in the visualization
    interp_batch_size = 8  # Number of gifs per row of visualization
    interp_z = vis_tools.sample_interp_zs(nz, interp_batch_size, interp_steps)
    interp_labels = np.zeros([interp_steps * interp_batch_size * nz, training_set.label_size], dtype=training_set.label_dtype)

    print('Setting up run dir...')
    misc.save_image_grid(grid_reals, os.path.join(submit_config.run_dir, 'reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    misc.save_image_grid(grid_fakes, os.path.join(submit_config.run_dir, 'fakes%06d.png' % resume_kimg), drange=drange_net, grid_size=grid_size)
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    summary_log.add_summary(build_image_summary(os.path.join(submit_config.run_dir, 'reals.png'), 'samples/real'), 0)
    summary_log.add_summary(build_image_summary(os.path.join(submit_config.run_dir, 'fakes%06d.png' % resume_kimg),
                                                'samples/Gs'), resume_kimg)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    if interp_snapshot_ticks != -1 and interp_snapshot_ticks < 9999:
        print('Generating initial interpolations...')
        vis_tools.make_and_save_interpolation_gifs(Gs, interp_z, interp_labels,
                                                   minibatch_size=sched.minibatch // submit_config.num_gpus,
                                                   interp_steps=interp_steps, interp_batch_size=interp_batch_size,
                                                   cur_kimg=resume_kimg, summary_log=summary_log)

    print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    num_G_grad_steps = 0

    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop():
            break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
        training_set.configure(sched.minibatch // submit_config.num_gpus, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for _mb_repeat in range(minibatch_repeats):
            for _D_repeat in range(D_repeats):
                tflib.run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch
            cur_hessian_weight = get_current_hessian_penalty_loss_weight(HP_args.hp_lambda, HP_args.hp_start_nimg,
                                                                         cur_nimg, HP_args.warmup_nimg)
            tflib.run([G_train_op] + reg_ops, {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch, hessian_weight: cur_hessian_weight})
            num_G_grad_steps += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d hessian_weight %s time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %-4.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch),
                autosummary('Progress/hessian_weight', cur_hessian_weight),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            autosummary('Progress/G_grad_steps', num_G_grad_steps)

            # Save snapshots and fake image samples (for both Gs and G):
            if cur_tick % image_snapshot_ticks == 0 or done:
                iter = (cur_nimg // 1000)
                grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch // submit_config.num_gpus)
                grid_fakes_inst = G.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch // submit_config.num_gpus)
                fake_path = os.path.join(submit_config.run_dir, 'fakes%06d.png' % iter)
                ifake_path = os.path.join(submit_config.run_dir, 'ifakes%06d.png' % iter)
                misc.save_image_grid(grid_fakes, fake_path, drange=drange_net, grid_size=grid_size)
                misc.save_image_grid(grid_fakes_inst, ifake_path, drange=drange_net, grid_size=grid_size)
                summary_log.add_summary(build_image_summary(fake_path, 'samples/Gs'), iter)
                summary_log.add_summary(build_image_summary(ifake_path, 'samples/G'), iter)

            # Generate/Save Interpolation Visualizations:
            if interp_snapshot_ticks != -1 and cur_tick % interp_snapshot_ticks == 0:
                vis_tools.make_and_save_interpolation_gifs(Gs, interp_z, interp_labels,
                                                           minibatch_size=sched.minibatch // submit_config.num_gpus,
                                                           interp_steps=interp_steps, interp_batch_size=interp_batch_size,
                                                           cur_kimg=cur_nimg // 1000, summary_log=summary_log)

            # Save snapshot and run metrics:
            if cur_tick % network_snapshot_ticks == 0 or done or cur_tick == 1:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                if cur_tick % network_metric_ticks == 0 or done or cur_tick == 1:
                    metrics.run(pkl, dataset_args=metrics_dataset_args, mirror_augment=mirror_augment,
                                num_gpus=submit_config.num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            ctx.update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % total_kimg))
    summary_log.close()

    ctx.close()

# ----------------------------------------------------------------------------
