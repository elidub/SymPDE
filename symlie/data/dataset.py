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

        self.augment = transform_kwargs['augment']

        self.space_length = data_kwargs['space_length']

        split_dir = os.path.join(data_dir, mode)
        data_kwargs_name = '_'.join([f'{k}={v}' for k, v in data_kwargs.items()])
        x = np.load(os.path.join(split_dir, f'x_{data_kwargs_name}.npy'))
        y = np.load(os.path.join(split_dir, f'y_{data_kwargs_name}.npy'))



        N = x.shape[0] if N == -1 else N

        self.x = torch.from_numpy(x[:N]).float()
        self.y = torch.from_numpy(y[:N]).float()
        
        if data_dir.split('/')[-1] in ['flower', 'sinev2']:
            centers = np.load(os.path.join(split_dir, f'centers_{data_kwargs_name}.npy'))
            self.centers = torch.from_numpy(centers[:N]).float()


    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augment == 'none':
            eps = torch.tensor([0.])
        elif self.augment == 'space_translation':
            eps = torch.rand((1,))
        elif self.augment == 'transform_flower':
            eps = torch.rand((4,))
            # x = x.view(self.space_length, self.space_length)[0]
            return x, y, eps, self.centers[index]
        else:
            raise NotImplementedError(f"Augmentation {self.augment} not implemented")
        
        return x, y, eps

    def __len__(self):
        return len(self.x)
