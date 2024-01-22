
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

def save_splits(create_sample_func: Callable, data_kwargs: dict, transform_kwargs: dict, data_dir: str, n_splits: dict = {'train': 400,'val': 1_000,'test': 1_000}) -> None:
    """Create and save train, val, and test splits of a dataset.

    Args:
        create_sample: Function that creates a sample from the dataset.
        n_splits: Dictionary with number of samples in each split.
        data_dir: Directory to save the splits.
    """
    for split, n_samples in zip(['train', 'val', 'test'], n_splits):
        print(f"Creating {n_samples} for {split}.")

        outs = create_sample_func(N = n_samples, **data_kwargs, **transform_kwargs)

        split_dir = os.path.join(data_dir, split) 
        os.makedirs(split_dir, exist_ok=True)

        data_kwargs_name = '_'.join([f'{k}={v}' for k, v in data_kwargs.items()])

        for k, v in outs.items():
            if type(v) == np.ndarray:
                np.save(os.path.join(split_dir, f'{k}_{data_kwargs_name}.npy'), v)
            elif type(v) == list:
                pickle.dump(v, open(os.path.join(split_dir, f'{k}_{data_kwargs_name}.pkl'), 'wb'))
            else:
                raise ValueError(f'Unupported type {type(v)}')