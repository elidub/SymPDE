import numpy as np
import sys, os
from matplotlib import pyplot as plt
import torch
from itertools import product
from tqdm import tqdm
import pickle
from typing import Callable, List, Union, Tuple
from PIL import Image

from data.transforms import Transform

def grid_1d(grid_size: int, x_min: float = 0., x_max: float = 1.):
    if isinstance(grid_size, tuple):
        grid_size_x, grid_size_y = grid_size
        assert grid_size_x == 1
        grid_size = grid_size_y

    x = np.linspace(x_min, x_max, grid_size, endpoint=False).reshape(-1, 1)
    return x
def grid_2d(grid_size: Union[int, tuple[int, int]], x_min: float = 0., x_max: float = 1.):

    if isinstance(grid_size, tuple):
        grid_size_x, grid_size_y = grid_size
        assert grid_size_x == grid_size_y
        grid_size = grid_size_x

    x = np.linspace(x_min, x_max, grid_size, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
    return xx, yy

def sine1d(N: int, y_low: int, y_high: int, grid_size: int):

    x = grid_1d(grid_size=grid_size, x_min=0, x_max=1)

    k = np.random.randint(y_low, y_high, size = (N,))

    sins = np.sin(2*np.pi*k*x).T

    return sins, k

def sine2d(N: int, y_low: int, y_high: int, grid_size: Union[int, tuple[int, int]]):

    xx, yy = grid_2d(grid_size=grid_size, x_min=0, x_max=1)

    k = np.random.randint(y_low, y_high, size = (N,))
    x_mult, y_mult = 0, 1

    sins = np.sin(2*np.pi*k*(xx*x_mult+yy*y_mult)).T

    return sins, k

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


class Create2dData:
    def __init__(self, create_sample_func: str, grid_size: tuple[int, ...], noise_std: float, y_low: int, y_high: int, eps_mult: List[float], only_flip: bool):
        self.create_sample_func = create_sample_func
        self.grid_size = grid_size #TODO: make naming consistent
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
        transform = Transform(grid_size=self.grid_size, eps_mult=self.eps_mult, only_flip=self.only_flip)
        centers_0 = torch.zeros(N, 2)
        epsilons  = torch.rand(N, 4)
        x = torch.from_numpy(x).reshape(N, np.prod(self.grid_size))
        x, centers = transform(x, centers=centers_0, epsilons=epsilons, transform_individual_bool=True)
        x = x.reshape(N, *self.grid_size)
        x, centers = x.numpy(), centers.numpy()
        return x, centers

    def __call__(self, N: int):
        x, y = self.create_sample_func(N, self.y_low, self.y_high, self.grid_size)

        x = self.add_noise(x)

        x, centers = self.transform_data(x, N)

        x = x.reshape(N, np.prod(self.grid_size))

        return {'x': x, 'y':y, 'centers': centers}

