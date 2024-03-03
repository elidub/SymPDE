from torch.utils.data import Dataset
import os
import pytorch_lightning as pl

import numpy as np
import torch

class FlatDataset(Dataset):

    def __init__(
            self, 
            mode: str, 
            task: str,
            data_params: dict, 
            data_vars: dict,
            transform_params: dict,
            data_dir = '../data/flat',
            N: int = -1,
        ):
        assert mode in ['train', 'test', 'val']

        if 'mnist' in data_dir:
            data_dir = os.path.join(data_dir[:-6], 'MNIST')

        split_dir = os.path.join(data_dir, mode)
        data_params_name = '_'.join([f'{k}={v}' for k, v in data_params.items()])
        data_vars_name   = '_'.join([f'{k}={v}' for k, v in data_vars.items()])
        transform_params_name = '_'.join([f'{k}={v}' for k, v in transform_params.items()])

        data = {d : np.load(os.path.join(split_dir, f'{d}_{data_params_name}_{data_vars_name}_{transform_params_name}.npy')) for d in ['x', 'y']}
        x, y = data['x'], data['y']

        N = x.shape[0] if N == -1 else N

        self.x = torch.from_numpy(x[:N]).float()
        self.y = torch.from_numpy(y[:N]).float()

        if task == 'classification':
            assert len(y.shape) == 1
            y_min = y.min()
            self.y = self.y - y_min # Shift classes such that they correspond for CrossEntropyLoss
            self.y = self.y.long()

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        centers = self.centers[index]
        return x, y, centers

    def __len__(self):
        return len(self.x)
