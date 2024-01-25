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

def grid_2d(grid_size: int, x_min: float = 0., x_max: float = 0.):
    x = np.linspace(x_min, x_max, grid_size)
    xx, yy = np.meshgrid(x, x)
    xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
    return xx, yy

def flower(N: int, y_low: int, y_high: int, grid_size: int, size: float = 3.):

    xx, yy = grid_2d(grid_size=grid_size, x_min=-size, x_max=size)

    s = 1
    n_leaves = np.random.randint(y_low, y_high, size = (N,))

    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # Defining a varying radius with a sine wave
    varying_radius = s + np.sin(n_leaves * theta)

    # Flower shape formula with varying extent
    z = (r <= varying_radius).T

    return z, n_leaves

def sine2d(N: int, y_low: int, y_high: int, grid_size: int):

    xx, yy = grid_2d(grid_size=grid_size, x_min=0, x_max=1)

    k = np.random.randint(y_low, y_high, size = (N,))
    x_mult, y_mult = 0, 1

    sins = np.sin(2*np.pi*k*(xx*x_mult+yy*y_mult)).T

    return sins, k

class Create2dData:
    def __init__(self, create_sample_func: str, space_length: int, noise_std: float, y_low: int, y_high: int, eps_mult: List[float], only_flip: bool):
        self.create_sample_func = create_sample_func
        self.grid_size = space_length #TODO: make naming consistent
        self.noise_std = noise_std
        self.y_low = y_low
        self.y_high = y_high
        self.eps_mult = eps_mult
        self.only_flip = only_flip

    def add_noise(self, x):
        noise = np.random.normal(0, self.noise_std, size = (x.shape))
        x = x + noise
        return x
    
    def transform_data(self, x, N):
        transform = Transform(only_flip=self.only_flip)
        centers_0 = torch.zeros(N, 2)
        epsilons  = torch.rand(N, 4)
        epsilons = epsilons * torch.tensor(self.eps_mult)
        x = torch.from_numpy(x).reshape(N, self.grid_size**2)
        x, centers = transform(x, centers=centers_0, epsilons=epsilons, transform_individual_bool=True)
        x = x.reshape(N, self.grid_size, self.grid_size)
        x, centers = x.numpy(), centers.numpy()
        return x, centers

    def __call__(self, N: int):
        x, y = self.create_sample_func(N, self.y_low, self.y_high, self.grid_size)

        x = self.add_noise(x)

        x, centers = self.transform_data(x, N)

        x = x.reshape(N, self.grid_size**2)

        # z = z.reshape(N, self.grid_size, self.grid_size)
        # z = z[:, 0, :]
        # z = z.reshape(N, self.grid_size) 

        return {'x': x, 'y':y, 'centers': centers}

# def create_flower(N: int, space_length: int, noise_std: float, y_low: int, y_high: int, eps_mult: List[float], only_flip: bool):

    
#     # Flower
#     z, target = flower(N, y_low, y_high, grid_size=space_length, size=3)
  
#     # Sine v2
#     z, target = sine(N, y_low, y_high,grid_size=space_length)

#     # Add noise
#     noise = np.random.normal(0, noise_std, size = (z.shape))
#     z = z + noise

#     # Transform
#     transform = Transform(only_flip=only_flip)
#     centers_0 = torch.zeros(N, 2)
#     epsilons  = torch.rand(N, 4)
#     epsilons = epsilons * torch.tensor(eps_mult)
#     z = torch.from_numpy(z).reshape(N, space_length**2)
#     z, centers = transform(z, centers=centers_0, epsilons=epsilons, transform_individual_bool=True)
#     z = z.reshape(N, space_length, space_length)
#     z, centers = z.numpy(), centers.numpy()

#     z = z.reshape(N, space_length**2)

#     # z = z.reshape(N, space_length, space_length)
#     # z = z[:, 0, :]
#     # z = z.reshape(N, space_length)

#     return {'x': z, 'y':target, 'centers': centers}

