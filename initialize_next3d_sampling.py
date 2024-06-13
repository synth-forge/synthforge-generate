import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import json
import matplotlib.pyplot as plt


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training_avatar_texture.triplane_next3d import TriPlaneGenerator

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def load_next3d():
    # Network pickle filename
    network_pkl = 'pretrained_models/next3d_ffhq_512.pkl'

    # List of random seeds
    seeds = [1]

    # Path of obj file
    obj_path = './data/demo/demo.obj'

    # Path of landmark file
    lms_path = './data/demo/demo_kpt2d.txt'

    # Truncation psi
    truncation_psi = 0.7

    # Truncation cutoff
    truncation_cutoff = 14

    # Where to save the output images
    outdir = './out'

    # Export shapes as .mrc files viewable in ChimeraX
    shapes = False

    # Shape resolution
    shape_res = 512

    # Field of View of camera in degrees
    fov_deg = 20#18.837

    # Shape Format
    shape_format = '.mrc'

    # If condition 2d landmarks?
    lms_cond = True

    # Overload persistent modules?
    reload_modules = True

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        print("Reloaded the modules")
    
    return G

def draw_landmarks(landmarks, color_fan=False):
    # Indices are based on the Dlib 68-point model
    segments = [
        list(range(0, 17)),  # Jawline
        list(range(17, 22)),  # Right eyebrow
        list(range(22, 27)),  # Left eyebrow
        list(range(27, 36)),  # Nose
        list(range(36, 42)),  # Right eye
        list(range(42, 48)),  # Left eye
        list(range(48, 61)),  # Outer lip
        list(range(61, 68))   # Inner lip
    ]
    
    if color_fan:
        color_pts = 'green'
        color_lines = 'green'
    else:
        color_pts = 'red'
        color_lines = 'blue'
    
    for segment in segments:
        # Loop through each segment and draw lines between adjacent landmarks
        for i in range(len(segment) - 1):
            plt.plot(
                [landmarks[segment[i], 0], landmarks[segment[i + 1], 0]],
                [landmarks[segment[i], 1], landmarks[segment[i + 1], 1]],
                color=color_lines,
                linewidth=1,
            )
    # close the looping points
    # Nose
    plt.plot(
        [landmarks[segments[3][-1], 0], landmarks[segments[3][3], 0]],
        [landmarks[segments[3][-1], 1], landmarks[segments[3][3], 1]],
        color=color_lines,
        linewidth=1,
    )
    # Eyes
    plt.plot(
        [landmarks[segments[4][-1], 0], landmarks[segments[4][0], 0]],
        [landmarks[segments[4][-1], 1], landmarks[segments[4][0], 1]],
        color=color_lines,
        linewidth=1,
    )
    plt.plot(
        [landmarks[segments[5][-1], 0], landmarks[segments[5][0], 0]],
        [landmarks[segments[5][-1], 1], landmarks[segments[5][0], 1]],
        color=color_lines,
        linewidth=1,
    )
    
    # Lips
    plt.plot(
        [landmarks[segments[6][-1], 0], landmarks[segments[7][0], 0]],
        [landmarks[segments[6][-1], 1], landmarks[segments[7][0], 1]],
        color=color_lines,
        linewidth=1,
    )
    plt.plot(
        [landmarks[segments[6][-1], 0], landmarks[segments[7][-1], 0]],
        [landmarks[segments[6][-1], 1], landmarks[segments[7][-1], 1]],
        color=color_lines,
        linewidth=1,
    )

    plt.scatter(landmarks[:, 0], landmarks[:, 1], color=color_pts, s=5)  # Draw the landmarks as red points