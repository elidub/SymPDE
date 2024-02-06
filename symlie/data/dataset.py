from torch.utils.data import Dataset
import os
import pytorch_lightning as pl

import numpy as np
import torch

class FlatDataset(Dataset):

    def __init__(
            self, 
            mode: str, 
            data_kwargs: dict, 
            transform_kwargs: dict,
            data_dir = '../data/flat',
            N: int = -1,
        ):
        assert mode in ['train', 'test', 'val']

        split_dir = os.path.join(data_dir, mode)
        data_kwargs_name = '_'.join([f'{k}={v}' for k, v in data_kwargs.items()])
        transform_kwargs_name = '_'.join([f'{k}={v}' for k, v in transform_kwargs.items()])

        try:
            data = {d : np.load(os.path.join(split_dir, f'{d}_{data_kwargs_name}_{transform_kwargs_name}.npy')) for d in ['x', 'y', 'centers']}
        except:
            data = {d : np.load(os.path.join(split_dir, f'{d}_{data_kwargs_name}.npy')) for d in ['x', 'y', 'centers']}
        x, y, centers = data['x'], data['y'], data['centers']

        N = x.shape[0] if N == -1 else N

        self.x = torch.from_numpy(x[:N]).float()
        self.y = torch.from_numpy(y[:N]).float()

        if 'MNIST' in data_dir:
            self.y = self.y.long()

        self.centers = torch.from_numpy(centers[:N])


    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        centers = self.centers[index]
        return x, y, centers

    def __len__(self):
        return len(self.x)
