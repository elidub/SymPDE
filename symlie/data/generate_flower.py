import numpy as np
import sys, os
from matplotlib import pyplot as plt
import torch
from itertools import product
from tqdm import tqdm
import pickle
from typing import Callable, List
from PIL import Image

from data.transforms import Transform

# from numpy import Array

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

def flower_grid(space_length: int, size: float = 3):
    # Define the dimensions of the grid
    x = np.linspace(-size, size, space_length)
    xx, yy = np.meshgrid(x, x)
    xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
    return xx, yy
def flower(xx, yy, N: int, y_low: int, y_high: int):
    s = 1
    n_leaves = np.random.randint(y_low, y_high, size = (N,))

    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # Defining a varying radius with a sine wave
    varying_radius = s + np.sin(n_leaves * theta)

    # Flower shape formula with varying extent
    z = (r <= varying_radius).T

    return z, n_leaves

# def sine_grid(space_length: int, size: float = 1):
#     x = np.linspace(0, 1, space_length).reshape(-1, 1)
#     xx, yy = np.meshgrid(x, x)
#     xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
#     return xx, yy    
# def sine(xx, yy, N: int, y_low: int, y_high: int):
#     k = np.random.randint(y_low, y_high, size = (N,))

#     # kx, ky = k, k
#     kx, ky = 0, 0

#     sins = np.sin(2*np.pi*(kx*xx+ky*yy)).T
#     return sins, k

def sine_grid(space_length: int, size: float = 1):
    x = np.linspace(0, 1, space_length).reshape(-1, 1)
    xx, yy = np.meshgrid(x, x)
    xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
    return xx, yy    
def sine(xx, yy, N: int, y_low: int, y_high: int):
    # k = np.random.randint(y_low, y_high, size = (N,))
    # sins = np.sin(2*np.pi*k*(xx+yy)).T

    k = np.random.randint(y_low, y_high, size = (N,))
    x_mult, y_mult = 0, 1

    sins = np.sin(2*np.pi*k*(xx*x_mult+yy*y_mult)).T

    return sins, k


def create_flower(N: int, space_length: int, noise_std: float, y_low: int, y_high: int, eps_mult: List[float], only_flip: bool):

    
    # Flower
    # xx, yy = flower_grid(space_length)
    # z, target = flower(xx, yy, N, y_low, y_high)
  
    # Sine v2
    xx, yy = sine_grid(space_length)
    z, target = sine(xx, yy, N, y_low, y_high)

    # Add noise
    noise = np.random.normal(0, noise_std, size = (z.shape))
    z = z + noise

    # Transform
    transform = Transform(only_flip=only_flip)
    centers_0 = torch.zeros(N, 2)
    epsilons  = torch.rand(N, 4)
    epsilons = epsilons * torch.tensor(eps_mult)
    z = torch.from_numpy(z).reshape(N, space_length**2)
    z, centers = transform(z, centers=centers_0, epsilons=epsilons, transform_individual_bool=True)
    z = z.reshape(N, space_length, space_length)
    z, centers = z.numpy(), centers.numpy()

    z = z.reshape(N, space_length**2)

    # z = z.reshape(N, space_length, space_length)
    # z = z[:, 0, :]
    # z = z.reshape(N, space_length)

    return {'x': z, 'y':target, 'centers': centers}

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