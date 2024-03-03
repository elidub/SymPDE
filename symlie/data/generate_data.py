
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
from data.generate_2d import sine1d, sine2d, flower, mnist, noise
from misc.viz import plot1d, plot2d

def save_splits(create_sample_func: Callable, data_params: dict, data_vars: dict, transform_params: dict, data_dir: str, n_splits: dict = {'train': 400,'val': 1_000,'test': 1_000}) -> None:
    """Create and save train, val, and test splits of a dataset.

    Args:
        create_sample: Function that creates a sample from the dataset.
        n_splits: Dictionary with number of samples in each split.
        data_dir: Directory to save the splits.
    """
    create_data = Create2dData(create_sample_func, data_params, data_vars, transform_params)

    data_params_name = '_'.join([f'{k}={v}' for k, v in data_params.items()])
    data_vars_name   = '_'.join([f'{k}={v}' for k, v in data_vars.items()])
    transform_params_name = '_'.join([f'{k}={v}' for k, v in transform_params.items()])

    for split, n_samples in zip(['train', 'val', 'test'], n_splits):
        print(f"Creating {n_samples} for {split}.")

        outs = create_data(N = n_samples, split = split)

        split_dir = os.path.join(data_dir, split) 
        os.makedirs(split_dir, exist_ok=True)


        for k, v in outs.items():
            assert type(v) == np.ndarray, f"Expected numpy array, got type of {k} = {type(v)}"
            np.save(os.path.join(split_dir, f'{k}_{data_params_name}_{data_vars_name}_{transform_params_name}.npy'), v)


datasets = {
    'noise': {
        'create_sample_func' : noise,
        'plot_func'          : plot1d,
        'data_params'        : {'grid_size': [1, 7], 'noise_std': 1.},
        'transform_params'   : {'eps_mult': [0., 0., 0., 0.], 'only_flip': False},
        'data_params_show'   : {'grid_size': [1, 7], 'noise_std': 1.},
    },
    'sine1d': {
        'create_sample_func' : sine1d,
        'plot_func'          : plot1d,
        'data_params'        : {'grid_size': [1, 7], 'noise_std': 0.5},
        'data_params_show'   : {'grid_size': [1, 100], 'noise_std': 0.1},
        'data_vars'          : {'k_low': 1, 'k_high': 3},
        'transform_params'   : {'eps_mult': [0., 0., 1., 0.], 'only_flip': False},
    },
    'sine2d': {
        'create_sample_func' : sine2d,
        'plot_func'          : plot2d,
        'data_params'        : {'grid_size':[7,7], 'noise_std':0.5},
        'data_params_show'   : {'grid_size':[16, 16], 'noise_std':0.0},
        'data_vars'          : {'k_low': 1, 'k_high': 3},
        'transform_params'   : {'eps_mult':[0., 0., 1., 1.], 'only_flip' : False},
    },
    'sine2d-rot': {
        'create_sample_func' : sine2d,
        'plot_func'          : plot2d,
        'data_params'        : {'grid_size':[7,7], 'noise_std':0.1},
        'data_params_show'   : {'grid_size':[16, 16], 'noise_std':0.0},
        'data_vars'          : {'k_low': 1, 'k_high': 3},
        'transform_params'   : {'eps_mult':[0., 1., 1., 1.], 'only_flip' : False},
    },
    'flower': {
        'create_sample_func' : flower,
        'plot_func'          : plot2d,
        'data_params'        : {'grid_size':[10,10], 'noise_std':0.1},
        'data_params_show'   : {'grid_size':[50, 50], 'noise_std':0.00},
        'data_vars'          : {'k_low': 2, 'k_high': 7},
        'transform_params'   : {'eps_mult':[0., 1., 1., 1.], 'only_flip' : False},
    },
    'mnist': {
        'create_sample_func' : mnist,
        'plot_func'          : plot2d,
        'data_params'        : {'grid_size': [7,7], 'noise_std': 0.0},
        'data_params_show'   : {'grid_size': [18, 18], 'noise_std': 0.0},
        'transform_params'   : {'eps_mult':[0., 0., 1., 1.], 'only_flip' : False},
    },
    'mnist-noise': {
        'create_sample_func' : mnist,
        'plot_func'          : plot2d,
        'data_params'        : {'grid_size': [7,7], 'noise_std': 0.1},
        'data_params_show'   : {'grid_size': [18, 18], 'noise_std': 0.0},
        'transform_params'   : {'eps_mult':[0., 0., 1., 1.], 'only_flip' : False},
    },
    'mnist-noise-rot': {
        'create_sample_func' : mnist,
        'plot_func'          : plot2d,
        'data_params'        : {'grid_size': [7,7], 'noise_std': 0.1},
        'data_params_show'   : {'grid_size': [18, 18], 'noise_std': 0.0},
        'transform_params'   : {'eps_mult':[0., 1., 1., 1.], 'only_flip' : False},
    },

}


