"""
Helper functions for creating disentanglement videos during training for visualization purposes.
All of these functions are structured under the assumption that the fake images were generated
from latent noise produced by sample_interp_zs() (which has a very specific pattern; see the function for more info).
"""

import numpy as np
import imageio
from training.misc import save_image_grid, create_image_grid, adjust_dynamic_range
from dnnlib.tflib.autosummary import convert_tensor_to_gif_summary
import moviepy.editor
import os


def prepro_imgs(frames, drange=[0, 255], nchw_to_nhwc=True):
    """Converts NCHW frame tensor in float [-1, 1] to NHWC tensor in uint8 [0, 255]"""
    frames = adjust_dynamic_range(frames, [-1, 1], drange)
    frames = np.rint(frames).clip(drange[0], drange[1])  # The rounding is important!
    if nchw_to_nhwc:
        frames = np.transpose(frames, (0, 2, 3, 1)).astype(np.uint8)
    return frames


def normalize_latents(z, epsilon=1e-8):
    """Pixel norm implementation in NumPy."""
    return z / np.sqrt(np.mean(np.square(z), axis=1, keepdims=True) + epsilon)


def sample_interp_zs(nz, batch_size, interp_steps, extent=2, apply_norm=False):
    """
    Samples z vectors used for visualizing disentanglement.

    :param nz: Number of latent z components in G's input
    :param batch_size: Number of samples (columns) per-z component
    :param interp_steps: Number of frames per video (=granularity of interpolation)
    :param extent: How far to interpolate each z vector?
    :param apply_norm: Whether to apply pixel norm to the initial sampled z vectors


    :return: Tensor of shape (nz*batch_size*interp_steps, nz).

    The easiest/fastest way to understand this function is to just feed it some toy inputs:

    >>> sample_interp_zs(nz=2, batch_size=2, interp_steps=3, extent=2)
        array([[-2.        , -1.2941029 ],      # z0, column 0, frame 0
               [ 0.        , -1.2941029 ],      # z0, column 0, frame 1
               [ 2.        , -1.2941029 ],      # z0, column 0, frame 2
               [-2.        ,  1.31908108],      # z0, column 1, frame 0
               [ 0.        ,  1.31908108],      # z0, column 1, frame 1
               [ 2.        ,  1.31908108],      # z0, column 1, frame 2
               [ 0.35134748, -2.        ],      # z1, column 0, frame 0
               [ 0.35134748,  0.        ],      # z1, column 0, frame 1
               [ 0.35134748,  2.        ],      # z1, column 0, frame 2
               [ 0.26627722, -2.        ],      # z1, column 1, frame 0
               [ 0.26627722,  0.        ],      # z1, column 1, frame 1
               [ 0.26627722,  2.        ]])     # z1, column 1, frame 2
    """

    zs = np.random.randn(batch_size, nz)
    if apply_norm:
        assert zs.ndim == 2
        zs = normalize_latents(zs)
    zs = zs.reshape((1, batch_size, 1, nz))
    zs = np.tile(zs, (nz, 1, interp_steps, 1)).reshape((nz * batch_size * interp_steps, nz))

    # interpolate linearly between [-extent, +extent] for the z component being changed
    delta_elements = 2 * extent * np.ones(nz) / (interp_steps - 1)

    # TODO: Should vectorize this part to make it easier to read.
    for z in range(nz):
        for b in range(batch_size):
            for s in range(interp_steps):
                ix = z * batch_size * interp_steps + b * interp_steps + s
                zs[ix, z] = s * delta_elements[z] - extent
    return zs


def create_compositional_video(nz, interp_steps, extent=2, apply_norm=False):
    def add_zs(last_z, i):
        last_z = np.array(last_z)  # deepcopy
        zi_start = last_z[i]
        gap = extent - zi_start if zi_start <= 0 else -(extent + zi_start)
        z_out = []
        for t in range(interp_steps):
            last_z[i] += gap / (interp_steps - 1)
            z_out.append(np.array(last_z))
        return z_out

    z = sample_interp_zs(nz, 1, interp_steps, extent, apply_norm)
    z = z[-interp_steps:][::-1]
    z_out = add_zs(z[-1], 10)
    z_out.extend(add_zs(z_out[-1], 1))
    z_out.extend(add_zs(z_out[-1], 11))
    z_out.extend(add_zs(z_out[-1], 10))
    z_out.extend(add_zs(z_out[-1], 1))
    z = np.stack([*z, *z_out], axis=0)
    return z


def create_interp_video_row(images, interp_steps, save_path=None, norm=False, px=0, py=0, transpose=False):
    """
    Creates (and optionally saves) a row showing side-by-side videos.
    We use this to generate a comparison of, e.g., interpolating a z component from -2 to +2
    for "batch_size" different samples of the fixed z components.

    :param images: Tensor of shape (interp_steps*batch_size, C, H, W) (rank 4 tensor)
    :param interp_steps: Number of frames in each video
    :param save_path: If specified, directly saves the video with the specified file path
    :param norm: If True, normalizes images before returning them (overrided by save_path)
    :return: If norm is True: (interp_steps, H, W * batch_size, C) tensor in uint8 [0, 255] (frames of video)
             If norm is False: (interp_steps, C, H, W * batch_size) tensor in float [-1, 1] (frames of video)
    """
    N = images.shape[0]
    bs = N // interp_steps
    assert N == bs * interp_steps
    frames = []
    grid_size = (1, bs) if transpose else (bs, 1)
    for i in range(interp_steps):  # Create each frame of the video:
        ixs = list(range(i, N, interp_steps))
        frame_i = create_image_grid(images[ixs], grid_size=grid_size, pad_val=-1, px=px, py=py)
        frames.append(frame_i)
    frames = np.stack(frames)
    if norm or save_path:  # Normalize the frames before returning?
        frames = prepro_imgs(frames)
    if save_path:  # Optionally directly save the frames as a video with 24fps
        imageio.mimsave(save_path, frames, duration=1/24)
    return frames


def create_multiple_interp_video_rows(images, interp_steps, batch_size, norm=True,
                                      save_dir=None, px=0, py=0, transpose=False):
    """
    Just a helper function that calls create_interp_video_row multiple times to generate a video
    for each of the z components.
    """
    nz = images.shape[0] // (interp_steps * batch_size)
    assert nz * interp_steps * batch_size == images.shape[0]
    offset = interp_steps * batch_size
    if save_dir:
        save_dir += '/z%03d.gif'
    row_gifs = []
    for z in range(nz):
        z_path = save_dir % z if save_dir else None
        z_gif = create_interp_video_row(images[z*offset:(z+1)*offset], interp_steps, norm=norm, save_path=z_path,
                                        px=px, py=py, transpose=transpose)
        row_gifs.append(z_gif)
    return row_gifs


def make_and_save_interpolation_gifs(G, interp_z, interp_labels, minibatch_size, interp_steps, interp_batch_size, cur_kimg, summary_log):
    """Generates the interpolation videos using G and interp_z. Then saves them to TensorBoard/ WandB."""
    interp_grid_fakes = G.run(interp_z, interp_labels, is_validation=True, minibatch_size=minibatch_size)
    interp_grid_fakes = create_multiple_interp_video_rows(interp_grid_fakes, interp_steps, interp_batch_size, norm=True)
    for grid_i, interp_grid in enumerate(interp_grid_fakes):  # Slightly adapted from TensorBoard issue 39:
        summ = convert_tensor_to_gif_summary(interp_grid, 'z/z%03d' % grid_i)
        summary_log.add_summary(summ, int(cur_kimg))


def save_video(frames, duration, fps, out_path):

    def make_frame(t):
        return frames.pop()

    moviepy.editor.VideoClip(make_frame, duration=duration).write_videofile(out_path, fps=fps, codec='libx264',
                                                                            bitrate='50M', threads=6,
                                                                            remove_temp=True,
                                                                            audio=False)


def make_h264_mp4_video(G, interp_z, interp_labels, minibatch_size, interp_steps, interp_batch_size, vis_path,
                        nz_per_vid=1, fps=60, perfect_loop=False, use_pixel_norm=True, n_frames=7,
                        stills_only=False, pad_x=0, pad_y=0, transpose=False):
    """
    Generates interpolation videos using G and interp_z, then saves them in "vis_path".
    This function returns a *much* higher quality video than make_and_save_interpolation_gifs.
    Additionally, if nz_per_vid > 1, this function "stitches" multiple z rows together to
    make visualization easier. The main entry point for using this function is visualize.py.
    """
    assert nz_per_vid > 0
    duration = (1 + perfect_loop) * interp_steps / fps
    nz = interp_z.shape[1]
    interp_grid_fakes = G.run(interp_z, interp_labels, is_validation=True,
                              minibatch_size=minibatch_size, normalize_latents=use_pixel_norm)
    if n_frames > 0:  # Save frames before we make the full video:
        save_flattened_frames(interp_grid_fakes, interp_steps, interp_batch_size, nz, n_frames, vis_path)
        if stills_only:
            return
    interp_grid_fakes = create_multiple_interp_video_rows(interp_grid_fakes, interp_steps, interp_batch_size,
                                                          norm=False, px=pad_x, py=pad_y//2, transpose=transpose)
    grid_size = (nz_per_vid, 1) if transpose else (1, nz_per_vid)
    print(f'Saving mp4 visualizations to {vis_path}.')
    for z_start in range(0, nz, nz_per_vid):  # Iterate over mp4s we are going to create:
        z_end = z_start + nz_per_vid
        row_videos = interp_grid_fakes[z_start: z_end]
        stitched_video = []  # Represents the "final" mp4 video we are going to save
        for i in range(interp_steps):  # iterate over frames
            row_frames = [row_video[i] for row_video in row_videos]
            row_frames = np.stack(row_frames)
            stitched_frame = create_image_grid(row_frames, grid_size=grid_size, px=0, py=pad_y//2, pad_val=-1)
            stitched_video.append(stitched_frame)
        if perfect_loop:  # Add a reversed copy of the video onto the end so it "loops"
            stitched_video = stitched_video + stitched_video[::-1]
        stitched_video.append(stitched_video[-1])  # Extra frame at the end
        stitched_video = prepro_imgs(np.asarray(stitched_video))  # Convert to np for normalization
        stitched_video = [frame for frame in stitched_video]  # Convert back to list for making the video
        video_path = os.path.join(vis_path, f'z{z_start:03}_to_z{z_end:03}.mp4')
        save_video(stitched_video, duration, fps, video_path)
    print(f'Done. Visualizations can be found in {vis_path}.')


def save_flattened_frames(images, interp_steps, samples, nz, n_frames, vis_path):
    """
    Same flattened versions of the disentanglement video as images.
    In each grid saved, moving across a row corresponds to interpolating a z component.
    Different rows re-sample the non-fixed z components.
    """
    from torchvision.utils import save_image
    import torch
    frames_path = os.path.join(vis_path, 'flattened_frames')
    os.makedirs(frames_path, exist_ok=True)
    images = images.reshape((nz, samples, interp_steps, *images.shape[1:]))
    keep_ixs = np.round(np.linspace(0, interp_steps - 1, num=n_frames)).astype(np.int32)
    keep_images = images[:, :, keep_ixs]
    for z_i, image_batch in enumerate(keep_images):
        image_batch = image_batch.reshape((samples * n_frames, *images.shape[3:]))
        # save_image_grid(image_batch, os.path.join(frames_path, f'z{z_i:03}.png'), grid_size=(n_frames, samples),
        #                 drange=[-1, 1])
        image_batch = (image_batch + 1) / 2  # [-1, 1] --> [0, 1] for torchvision
        save_image(torch.tensor(image_batch), os.path.join(frames_path, f'z{z_i:03}.png'), nrow=n_frames)
