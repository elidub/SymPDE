import numpy as np
import sys, os
from matplotlib import pyplot as plt
import torch
from itertools import product
from tqdm import tqdm
import pickle
from typing import Callable
from PIL import Image

from data.transforms import Transform

# from torchvision.transforms import Compose, RandomRotation, CenterCrop, Pad, ToTensor, Resize, RandomCrop
# from torchvision.transforms import RandomAffine, InterpolationMode

# from data.transforms import ToArray, Slice, SqueezeTransform, BaseTransform, SpaceTranslate, RandomPad, RandomScale, CustomRandomRotation, UnsqueezeTransform, MyPad, MyPrint

# def transform(x, centers, epsilons = [1., 1., 1., 1.], sample = True, antialias= False):
#     _, w, h = x.shape
#     assert w == h
#     space_length = w
#     return CustomCompose(
#         [
#             # Scaling
#             RandomScale(eps=epsilons[0], sample=sample),
            
#             # Rotation, scaling to supress aliasing
#             Resize(space_length*3, antialias=antialias),
#             CustomRandomRotation(eps=epsilons[1], sample=sample, interpolation=InterpolationMode.BILINEAR),
#             Resize(space_length, antialias=antialias),
            
#             # Space translation
#             SpaceTranslate(eps=epsilons[2], dim = 1, sample=sample), # x translation
#             SpaceTranslate(eps=epsilons[3], dim = 2, sample=sample), # y translation
#         ], 
#         centers = centers
#     )(x)

# def transform_batch(x, centers, epsilons=[1.,1.,1.,1.], sample=True):
#     xs, centers = zip(*[transform(x_i, centers=centers_i, epsilons=epsilons, sample=sample) for x_i, centers_i in tqdm(zip(x.unsqueeze(1), centers), total = len(x), leave=False)])
#     x, centers = torch.cat(xs), torch.cat(centers)
#     return x, centers

def create_flower(N: int, space_length: int, noise_std: float, y_low: int, y_high: int):
    
    sqrt = np.sqrt(space_length)
    space_length = int(sqrt)
    assert space_length == sqrt

    size = 3

    # Define the dimensions of the grid
    x = np.linspace(-size, size, space_length)
    xx, yy = np.meshgrid(x, x)

    # s = np.random.uniform(0.5, 1.5, size = (N,))
    s = 1
    n_leaves = np.random.randint(y_low, y_high, size = (N,))

    # Convert Cartesian to polar coordinates
    xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # Defining a varying radius with a sine wave
    varying_radius = s + np.sin(n_leaves * theta)

    # Flower shape formula with varying extent
    z = (r <= varying_radius).T

    # z = torch.from_numpy(z)
    # return z, centers_0

    # Transform
    transform = Transform()
    centers_0 = torch.zeros(N, 2)
    z, centers = transform(torch.from_numpy(z), centers = centers_0,  transform_individual_bool=True)
    z, centers = z.numpy(), centers.numpy()

    # Add noise
    noise = np.random.normal(0, noise_std, size = (z.shape))
    z_noise = z + noise
    z_noise = z_noise.reshape(N, space_length**2)

    return {'x': z_noise, 'y':n_leaves, 'centers': centers}

def plot_flower(x, y = None, l = 1, set_axis_off = True):
    N_plot = len(x)
    fig, axs = plt.subplots(1, N_plot, figsize = (N_plot*l, l), sharey=True, sharex=True)
    if N_plot == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(x[i], aspect='auto')
        if set_axis_off: ax.set_axis_off()
        if y is not None: axs[i].set_title(f'y = {y[i]}')
    plt.show()