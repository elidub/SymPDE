
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

from symlie.data.generate_2d import create_flower
from run import process_args

def save_splits(create_sample_func: Callable, data_kwargs: dict, data_dir: str, n_splits: dict = {'train': 400,'val': 1_000,'test': 1_000}) -> None:
    """Create and save train, val, and test splits of a dataset.

    Args:
        create_sample: Function that creates a sample from the dataset.
        n_splits: Dictionary with number of samples in each split.
        data_dir: Directory to save the splits.
    """
    for split, n_samples in zip(['train', 'val', 'test'], n_splits):
        print(f"Creating {n_samples} for {split}.")

        outs = create_sample_func(N = n_samples, **data_kwargs)

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


def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='Generating PDE data')

    parser.add_argument("--data_dir", type=str, default="../data/test", help="Path to data directory")
    parser.add_argument("--data_name", type=str, default="flower", help="Name of dataset")
    parser.add_argument("--data_kwargs", help="Data kwargs", type=json.loads)

    parser.add_argument("--n_splits", default=[400,1_000,1_000], nargs='+', help="Train, val, test split")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    args = process_args(args)

    create_sample_func_dict = {
        'flower': create_flower
    }

    save_splits(
        create_sample_func=create_sample_func_dict[args.data_name],
        data_kwargs=args.data_kwargs,
        data_dir=args.data_dir,
        n_splits=args.n_splits
    )


if __name__ == '__main__':
    args = parse_options()
    print(args)
    main(args)