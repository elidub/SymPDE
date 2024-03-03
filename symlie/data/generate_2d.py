import numpy as np
import sys, os
from matplotlib import pyplot as plt
import torch
from itertools import product
from tqdm import tqdm
import pickle
from typing import Callable, List, Union, Tuple
from PIL import Image
from torchvision import datasets, transforms

from data.transforms import Transform

def grid_1d(grid_size: int, x_min: float = 0., x_max: float = 1.):
    if isinstance(grid_size, list): grid_size = tuple(grid_size)
    if isinstance(grid_size, tuple):
        grid_size_x, grid_size_y = grid_size
        assert grid_size_x == 1
        grid_size = grid_size_y

    x = np.linspace(x_min, x_max, grid_size, endpoint=False).reshape(-1, 1)
    return x
def grid_2d(grid_size: Union[int, tuple[int, int]], x_min: float = 0., x_max: float = 1.):
    if isinstance(grid_size, list): grid_size = tuple(grid_size)
    if isinstance(grid_size, tuple):
        grid_size_x, grid_size_y = grid_size
        assert grid_size_x == grid_size_y
        grid_size = grid_size_x

    x = np.linspace(x_min, x_max, grid_size, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
    return xx, yy


def noise(N: int, grid_size: int):

    if isinstance(grid_size, tuple):
        grid_size_x, grid_size_y = grid_size
        if grid_size_x == 1:
            zeros = np.zeros((N, grid_size_y))
        else:
            zeros = np.zeros((N, *grid_size))
    elif isinstance(grid_size, int):
        zeros = np.zeros((N, grid_size))
    else:
        raise ValueError(f"grid_size = {grid_size} not understood.")

    random = np.random.normal(size=N)
    return zeros, random

# def sine1d(N: int, k_low: int, k_high: int, grid_size: int):
def sine1d(N: int, grid_size: int, k_low: int, k_high: int, A_low: float = 1., A_high: float = 1.):
    
    x = grid_1d(grid_size=grid_size, x_min=0, x_max=1)

    k = np.random.randint(k_low, k_high, size = (N,))
    A = np.random.uniform(A_low, A_high, size = (N))

    sins = np.reshape(A, (-1, 1))*np.sin(2*np.pi*k*x).T

    y = np.stack([k, A], axis = 1)

    return sins, np.stack([k, A], axis = 1)

def sine2d(N: int, grid_size: Union[int, tuple[int, int]], k_low: int, k_high: int, ):

    xx, yy = grid_2d(grid_size=grid_size, x_min=0, x_max=1)

    k = np.random.randint(k_low, k_high, size = (N,))
    x_mult, y_mult = 0, 1

    sins = np.sin(2*np.pi*k*(xx*x_mult+yy*y_mult)).T

    return sins, k

def flower(N: int, k_low: int, k_high: int, grid_size: int, size: float = 3., noise_std: float = 0.):

    xx, yy = grid_2d(grid_size=grid_size, x_min=-size, x_max=size)
    print(xx.shape, yy.shape)

    s = 1
    n_leaves = np.random.randint(k_low, k_high, size = (N,))

    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # Defining a varying radius with a sine wave
    varying_radius = s + np.sin(n_leaves * theta)

    # Flower shape formula with varying extent
    z = (r <= varying_radius).T

    return z, n_leaves

def mnist(N: int, grid_size: tuple[int, int], split: str):
    data_dir = '../data'
    train = {'train': True, 'val': True, 'test':False}[split]
    idx   = {'train': 0, 'val': 50_000, 'test': 0}

    # check if data exists
    if not os.path.exists(os.path.join(data_dir, 'MNIST/raw')):
        datasets.MNIST(data_dir, train=True, download=True)
        datasets.MNIST(data_dir, train=False, download=True)
    dataset = datasets.MNIST(data_dir, train=train, download=False)

    x = dataset.data
    y = dataset.targets
    
    # compose dataset
    compose = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(grid_size, antialias=True)
    ])
    x = compose(x / 255.)


    i = idx[split]
    assert i + N < len(dataset), f"{i+N} >= {len(dataset)}"
    if split == 'train':
        assert i + N < idx['val'] , f"{i+N} >= {idx['val']}"

    x = x[i:i+N].numpy()
    y = y[i:i+N].numpy()
    return x, y


class Create2dData:
    def __init__(self, create_sample_func: str, data_params: dict, data_vars: dict, transform_params: dict):
        self.create_sample_func = create_sample_func
        self.grid_size, self.noise_std = data_params['grid_size'], data_params['noise_std']
        self.data_vars = data_vars
        self.transform_params = transform_params

    def add_noise(self, x):
        noise = np.random.normal(0, self.noise_std, size = (x.shape))
        x = x + noise
        return x
    
    def transform_data(self, x, N):
        transform = Transform(grid_size=self.grid_size **self.transform_params)
        epsilons  = torch.rand(N, 4)
        x = torch.from_numpy(x).reshape(N, np.prod(self.grid_size))
        x = transform(x, epsilons=epsilons, transform_individual_bool=True)
        x = x.reshape(N, *self.grid_size).numpy()
        return x

    def __call__(self, N: int, split: str = 'train'):


        data_vars = self.data_vars.copy()
        if self.create_sample_func == mnist: data_vars['split'] = split

        x, y = self.create_sample_func(N, **data_vars)

        x = self.add_noise(x)

        x = self.transform_data(x, N)

        return {'x': x, 'y':y}

