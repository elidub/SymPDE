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
            args, 
            data_kwargs: dict, 
            transform_kwargs: dict,
            data_dir = '../data/flat',
            N: int = -1,
        ):
        assert mode in ['train', 'test', 'val']

        if 'mnist' in data_dir:
            data_dir = os.path.join(data_dir[:-6], 'MNIST')

        transform_kwargs['eps_mult'] = list(transform_kwargs['eps_mult'])

        split_dir = os.path.join(data_dir, mode)
        data_kwargs_name = '_'.join([f'{k}={v}' for k, v in data_kwargs.items()])
        transform_kwargs_name = '_'.join([f'{k}={v}' for k, v in transform_kwargs.items()])

        try:
            data = {d : np.load(os.path.join(split_dir, f'{d}_{data_kwargs_name}_{transform_kwargs_name}.npy')) for d in ['x', 'y', 'centers']}
        except:
            try:
                data = {d : np.load(os.path.join(split_dir, f'{d}_{data_kwargs_name}.npy')) for d in ['x', 'y', 'centers']}
            except:
                try:
                    transform_kwargs['eps_mult'] = tuple(transform_kwargs['eps_mult'])
                    transform_kwargs_name = '_'.join([f'{k}={v}' for k, v in transform_kwargs.items()])
                    data = {d : np.load(os.path.join(split_dir, f'{d}_{data_kwargs_name}_{transform_kwargs_name}.npy')) for d in ['x', 'y', 'centers']}
                except:
                    raise ValueError(f"Data could not be loaded correctly! {data_kwargs_name}_{transform_kwargs_name}")




        x, y, centers = data['x'], data['y'], data['centers']

        N = x.shape[0] if N == -1 else N

        # y = y[:, 1]

        self.x = torch.from_numpy(x[:N]).float()
        self.y = torch.from_numpy(y[:N]).float()

        if task == 'classification':
            assert len(y.shape) == 1
            y_min = y.min()
            self.y = self.y - y_min # Shift classes such that they correspond for CrossEntropyLoss
            self.y = self.y.long()

        if hasattr(args, 'y_multi'):
            self.y = torch.stack([self.y*(1+y_range_i) for y_range_i in range(args.y_multi)]).T

        self.centers = torch.from_numpy(centers[:N])


    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        centers = self.centers[index]
        return x, y, centers

    def __len__(self):
        return len(self.x)
