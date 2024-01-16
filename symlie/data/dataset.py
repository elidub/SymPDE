from torch.utils.data import Dataset
import os
import pytorch_lightning as pl

import numpy as np
import torch

class FlatDataset(Dataset):

    def __init__(
            self, 
            mode: str, 
            data_kwargs: dict = {
                'space_length': 4,
                'noise_std': 0.1,
                'y_low': 0.5,
                'y_high': 2.,
            }, 
            transf_kwargs: dict = {
                'augment': 'none',
            },
            data_dir = '../data/flat',
            N: int = -1,
        ):
        assert mode in ['train', 'test', 'val']

        self.augment = transf_kwargs['augment']

        split_dir = os.path.join(data_dir, mode)
        data_kwargs_name = '_'.join([f'{k}={v}' for k, v in data_kwargs.items()])
        x = np.load(os.path.join(split_dir, f'x_{data_kwargs_name}.npy'))
        y = np.load(os.path.join(split_dir, f'y_{data_kwargs_name}.npy'))

        N = x.shape[0] if N == -1 else N

        self.x = torch.from_numpy(x[:N]).float()
        self.y = torch.from_numpy(y[:N]).float()


    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augment == 'none':
            eps = torch.tensor([0.])
        elif self.augment == 'space_translation':
            eps = torch.rand((1,))
        else:
            raise NotImplementedError
        
        return x, y, eps

    def __len__(self):
        return len(self.x)
