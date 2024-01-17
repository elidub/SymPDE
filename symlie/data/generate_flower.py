import numpy as np
import sys, os
from matplotlib import pyplot as plt
import torch
from itertools import product
from tqdm import tqdm
import pickle

from typing import Callable
from PIL import Image

from torchvision.transforms import Compose, RandomRotation, CenterCrop, Pad, ToTensor, Resize, RandomCrop
from torchvision.transforms import RandomAffine, InterpolationMode

from data.transforms import ToArray, Slice, SqueezeTransform, BaseTransform, SpaceTranslate, RandomPad, RandomScale, CustomRandomRotation, CustomCompose, UnsqueezeTransform, MyPad, MyPrint

def transform(x, augment_kwargs, epsilons = [1., 1., 1., 1.], sample = True, antialias= False):
    _, w, h = x.shape
    assert w == h
    space_length = w
    return CustomCompose(
        [
            # Scaling
            RandomScale(eps=epsilons[0], sample=sample),
            
            # Rotation, scaling to supress aliasing
            Resize(space_length*3, antialias=antialias),
            CustomRandomRotation(eps=epsilons[1], sample=sample, interpolation=InterpolationMode.BILINEAR),
            Resize(space_length, antialias=antialias),
            
            # Space translation
            SpaceTranslate(eps=epsilons[2], dim = 1, sample=sample), # x translation
            SpaceTranslate(eps=epsilons[3], dim = 2, sample=sample), # y translation
        ], 
        augment_kwargs = augment_kwargs
    )(x)

def transform_batch(x, augment_kwargs, epsilons=[1.,1.,1.,1.], sample=True):
    xs, augment_kwargs = zip(*[transform(x_i, augment_kwargs = augment_kwarg, epsilons=epsilons, sample=sample) for x_i, augment_kwarg in tqdm(zip(x.unsqueeze(1), augment_kwargs), total = len(x), leave=False)])
    x = torch.cat(xs)
    augment_kwargs = list(augment_kwargs)
    return x, augment_kwargs

def create_flower(N: int, space_length: int, noise_std: float, y_low: int, y_high: int):
    
    sqrt = np.sqrt(space_length)
    space_length = int(sqrt)
    assert space_length == sqrt

    size = 3
    augment_kwargs0 = {'SpaceTranslate_1' : 0.0, 'SpaceTranslate_2': 0.0}

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

    z, augment_kwargs = transform_batch(torch.from_numpy(z), augment_kwargs = [augment_kwargs0]*N)
    z = z.numpy()

    noise = np.random.normal(0, noise_std, size = (z.shape))

    z_noise = z + noise

    z_noise = z_noise.reshape(N, space_length**2)

    return {'x': z_noise, 'y':n_leaves, 'augment_kwargs': augment_kwargs}