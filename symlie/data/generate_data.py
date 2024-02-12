
import argparse
import numpy as np
import sys, os
from matplotlib import pyplot as plt
from itertools import product
from tqdm import tqdm
import pickle
import json
from typing import Callable
from PIL import Image

from data.generate_2d import Create2dData
from data.generate_2d import sine1d, sine2d, flower, mnist
from misc.viz import plot1d, plot2d

def save_splits(create_sample_func: Callable, data_kwargs: dict, transform_kwargs: dict, data_dir: str, n_splits: dict = {'train': 400,'val': 1_000,'test': 1_000}) -> None:
    """Create and save train, val, and test splits of a dataset.

    Args:
        create_sample: Function that creates a sample from the dataset.
        n_splits: Dictionary with number of samples in each split.
        data_dir: Directory to save the splits.
    """
    create_data = Create2dData(create_sample_func, data_kwargs, transform_kwargs)

    data_kwargs_name = '_'.join([f'{k}={v}' for k, v in data_kwargs.items()])
    transform_kwargs_name = '_'.join([f'{k}={v}' for k, v in transform_kwargs.items()])

    for split, n_samples in zip(['train', 'val', 'test'], n_splits):
        print(f"Creating {n_samples} for {split}.")

        outs = create_data(N = n_samples, split = split)

        split_dir = os.path.join(data_dir, split) 
        os.makedirs(split_dir, exist_ok=True)


        for k, v in outs.items():
            assert type(v) == np.ndarray, f"Expected numpy array, got type of {k} = {type(v)}"
            np.save(os.path.join(split_dir, f'{k}_{data_kwargs_name}_{transform_kwargs_name}.npy'), v)


datasets = {
    'sine1d': {
        'create_sample_func' : sine1d,
        'plot_func'          : plot1d,
        'data_kwargs'        : {'grid_size': (1, 7), 'noise_std': 0.5, 'y_low': 1, 'y_high': 3},
        'transform_kwargs'   : {'eps_mult': [0., 0., 1., 0.], 'only_flip': False},
        'data_kwargs_show'   : {'grid_size': (1, 100), 'noise_std': 0.1, 'y_low': 1, 'y_high': 3},
    },
    'sine2d': {
        'create_sample_func' : sine2d,
        'plot_func'          : plot2d,
        'data_kwargs'        : {'grid_size':(7, 7), 'noise_std':0.5, 'y_low':1, 'y_high':3},
        'transform_kwargs'   : {'eps_mult':[0., 0., 1., 1.], 'only_flip' : False},
        'data_kwargs_show'   : {'grid_size':(16, 16), 'noise_std':0.0, 'y_low':1, 'y_high':3},
    },
    'flower': {
        'create_sample_func' : flower,
        'plot_func'          : plot2d,
        'data_kwargs'        : NotImplemented,
        'transform_kwargs'   : NotImplemented,
        'data_kwargs_show'   : NotImplemented,
    },
    'mnist': {
        'create_sample_func' : mnist,
        'plot_func'          : plot2d,
        'data_kwargs'        : {'grid_size': (7,7), 'noise_std': 0.0},
        'transform_kwargs'   : {'eps_mult':[0., 0., 1., 1.], 'only_flip' : False},
        'data_kwargs_show'   : {'grid_size': (18, 18), 'noise_std': 0.0},
    },
}


