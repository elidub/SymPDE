import os
import torch
import numpy as np
import pytorch_lightning as pl

from data.utils import load_obj
from data.utils import d_to_coords
from data.pde_data_aug import augment_pde1, augment_KdV

class PDEDataset(torch.utils.data.Dataset):
    def __init__(
        self, pde_name, data_dir, split="val", 
        transform=None, target_transform=None,
        generators = None, n_samples = None
    ):
        self.split = split
        split_dir = os.path.join(data_dir, split)

        self.transform = transform
        self.target_transform = target_transform
        self.generators = generators

        self.us, self.dx, self.dt = load_obj(os.path.join(split_dir, f"{pde_name}"))
        
        if n_samples is not None:
            if n_samples > len(self.us):
                print(f'n_samples = {n_samples} > number of available samples = {len(self.us)}.\nReturning only all available samples!')
                n_samples = len(self.us)
            print(f'Selecting {n_samples} out of the {len(self.us)} {split} samples!')
            self.us = self.us[:n_samples]
        
        if pde_name == 'pde1':
            self.augment_pde = augment_pde1
        elif pde_name == 'KdV':
            self.augment_pde = augment_KdV
        else:
            raise ValueError(f'Augmenting of {pde_name} not implemented!')

        if self.generators is True:
            print(f'Augmenting {pde_name}!')

    def __len__(self):
        return len(self.us)
    
    def __getitem__(self, idx):

        u = self.us[idx]

        if self.transform:
            u = self.transform(u)

        if self.target_transform:
            u = self.target_transform(u)

        u = torch.from_numpy(u)

        u, dx, dt = u.float(), torch.tensor(self.dx, dtype = torch.float32), torch.tensor(self.dt, dtype = torch.float32)

        if self.generators is not None:
            u, dx, dt = self.augment(u, dx, dt)

        return u, dx, dt

    def augment(self, u, dx, dt):
        """
        Augment similar as LPSDA
        """
        # Get coordinates
        X = d_to_coords(u, dx, dt)
        x, t = X.permute(2, 0, 1)[:2]

        # # Augment
        # sol = (u, X)
        # for g in self.generators:
        #     sol = g(sol, shift = 'fourier')
        # u, X = sol

        # # Get new coordinates
        # dx = X[0, 1, 0] - X[0, 0, 0]
        # dt = X[1, 0, 1] - X[0, 0, 1]
        # print(u.shape, dx, dt)

        if self.generators is True:
            u, x, t = self.augment_pde(u.clone(), x.clone(), t.clone())
            dx = x[0,1] - x[0, 0]
            dt = t[1,0] - t[0, 0]

        return u, dx, dt
    


class PDEDataModule(pl.LightningDataModule):
    def __init__(self, pde_name, data_dir, 
                 batch_size=1, num_workers=1,
                 generators = None,
                 n_splits = [160, 20, 20],
                 persistent_workers = False,
                ):
        super().__init__()
        self.pde_name = pde_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.splits = ['train', 'val', 'test']
        
        self.generators = generators
        self.n_splits = {'train': n_splits[0], 'val': n_splits[1], 'test': n_splits[2]}
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        self.dataset = { 
            split : 
                PDEDataset(
                    pde_name=self.pde_name, 
                    data_dir=self.data_dir, 
                    split=split,
                    generators = self.generators if split == 'train' else None,
                    n_samples    = self.n_splits[split],
                ) 
            for split in self.splits }
        self.collate_fn = self.custom_collate

    def custom_collate(self, batch):
        us, dx, dt = zip(*batch)
        return torch.stack(us), torch.stack(dx), torch.stack(dt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )