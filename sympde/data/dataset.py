import os
import torch
import numpy as np
import pytorch_lightning as pl

from data.utils import load_obj
from data.utils import d_to_coords
from data.pde_data_aug import augment_pde1, augment_KdV

import os, sys
sys.path.append(os.path.join(os.getcwd(), '../ext_repos/LPSDA'))
from common.utils import HDF5Dataset, DataCreator
from equations.PDEs import PDE, KdV, KS, Heat

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

        # pde = KdV()
        # pde.max_velocity = 0.4
        # self.us = dataset_split = select_hdf5_dataset(split, pde)
        # if split == 'val': split = 'valid'
        # hdf5_data_dir = f'../ext_repos/LPSDA/data/KdV_{split}_easy.h5' if split != 'train' else f'../ext_repos/LPSDA/data/KdV_{split}_10_easy.h5' 
        # self.us = HDF5Dataset(hdf5_data_dir,
        #     mode=split,
        #     nt=140,
        #     nx=256,
        #     shift='fourier',
        #     pde=pde,
        # )

        self.us, self.dxs, self.dts = load_obj(os.path.join(split_dir, f"{pde_name}"))
        
        # if n_samples is not None:
        #     if n_samples > len(self.us):
        #         print(f'n_samples = {n_samples} > number of available samples = {len(self.us)}.\nReturning only all available samples!')
        #         n_samples = len(self.us)
        #     print(f'Selecting {n_samples} out of the {len(self.us)} {split} samples!')
        #     self.us = self.us[:n_samples]
        
        if pde_name == 'pde1':
            self.augment_pde = augment_pde1
        elif pde_name == 'KdV':
            self.augment_pde = augment_KdV
        else:
            raise ValueError(f'Augmenting of {pde_name} not implemented!')

        if self.generators is True:
            print(f'Augmenting {pde_name}!')
        else:
            print(f'Not augmenting {pde_name}!')

    def __len__(self):
        return len(self.us)
    
    def __getitem__(self, idx):

        # u = self.us[idx]
        u, dx, dt = self.us[idx], self.dxs[idx], self.dts[idx]

        if self.transform:
            u = self.transform(u)

        if self.target_transform:
            u = self.target_transform(u)

        u = torch.tensor(u)
        dx = torch.tensor(dx)
        dt = torch.tensor(dt)

        # u, dx, dt = u.float(), torch.tensor(self.dxs[idx], dtype = torch.float32), torch.tensor(self.dts[idx], dtype = torch.float32)


        if self.generators is True:
            u, dx, dt = self.augment(u, dx, dt)

        return u.float(), dx.float(), dt.float()

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

        u, x, t = self.augment_pde(u.clone(), x.clone(), t.clone())
        dx = x[0,1] - x[0, 0]
        dt = t[1,0] - t[0, 0]
        
        # return u, dx, dt
        return u, dx, dt
    
def select_hdf5_dataset(split, pde):
    if split == 'val': split = 'valid'
    # path = '../ext_repos/LPSDA/data'
    path = 'data'
    hdf5_data_dir = f'{path}/KdV_{split}_easy.h5' if split != 'train' else f'{path}/KdV_{split}_10_easy.h5' 
    dataset_split = HDF5Dataset(
        hdf5_data_dir,
        mode=split,
        nt=140,
        nx=256,
        shift='fourier',
        pde=pde,
    )
    return dataset_split

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

        pde = KdV()
        if self.generators is True:
            print(f'Augmenting {self.pde_name}!')
            pde.max_velocity = 0.4
        else:
            print(f'Not augmenting {self.pde_name}!')
        
        self.dataset = { 
            split : 
                PDEDataset(
                    pde_name=self.pde_name, 
                    data_dir=self.data_dir, 
                    split=split,
                    generators = self.generators if split == 'train' else None,
                    n_samples    = self.n_splits[split],
                ) 
                # select_hdf5_dataset(split, pde)
            for split in self.splits 
        }
        self.collate_fn = self.custom_collate

    def custom_collate(self, batch):
        us, dx, dt = zip(*batch)
        return torch.stack(us), torch.stack(dx), torch.stack(dt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, #TODO: shuffle=True
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