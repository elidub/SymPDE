import os
import torch
import numpy as np
import pytorch_lightning as pl

from misc.utils import load_obj
from data.utils import d_to_coords
from data.pdes import PDEs

class PDEDataset(torch.utils.data.Dataset):
    def __init__(
        self, pde_name, data_dir, split="val", 
        transform=None, target_transform=None,
        epsilons=[], n_samples = -1
    ):
        self.split = split

        self.transform = transform
        self.target_transform = target_transform
        self.epsilons = epsilons

        self.pde = PDEs()[pde_name]
        self.us, self.dxs, self.dts = load_obj(os.path.join(data_dir, split, f"{pde_name}"))
        
        n_samples = len(self.us) if n_samples == -1 else n_samples
        if n_samples > len(self.us):
            print(f'n_samples = {n_samples} > number of available samples = {len(self.us)}.\nReturning only all available samples!')
            n_samples = len(self.us)
        print(f'Selecting {n_samples} out of the {len(self.us)} {split} samples!')
        self.us = self.us[:n_samples]

        if self.epsilons:
            print(f'Augmenting {pde_name} with epsilons {self.epsilons}!')
            assert len(self.pde.aug_methods) == len(self.epsilons), f'Number of epsilons ({len(self.epsilons)}) must match number of augmentations ({self.pde.aug_methods})'

    def __len__(self):
        return len(self.us)
    
    def __getitem__(self, idx):

        u, dx, dt = self.us[idx], self.dxs[idx], self.dts[idx]

        # i, j = 40, 40
        # u = u[:i, :j]

        u = torch.from_numpy(u)
        dx = torch.tensor(dx)
        dt = torch.tensor(dt)

        if self.transform:
            u = self.transform(u)

        if self.target_transform:
            u = self.target_transform(u)

        # torch.float64 are used for augmentation
        if self.epsilons:
            u, dx, dt = self.augment(u, dx, dt)

        # torch.float32 are passed to the model
        u, dx, dt = u.float(), dx.float(), dt.float()
        return u, dx, dt

    def augment(self, u, dx, dt, epsilons = None, rand = True):
        """
        Augment similar as LPSDA
        """
        # Get coordinates
        X = d_to_coords(u, dx, dt)
        x, t = X.permute(2, 0, 1)[:2]

        epsilons = self.epsilons if epsilons is None else epsilons

        # Augment
        # u, x, t = self.pde.augment(u.clone(), x.clone(), t.clone(), epsilons=epsilons)
        for aug_method, epsilon in zip(self.pde.aug_methods, epsilons):
            if epsilon > 0:
                eps = epsilon * (torch.rand(()) - 0.5) if rand else torch.tensor([epsilon])
                # print(f'Augmenting with {aug_method} with epsilon = {eps}')
                u, x, t = aug_method(u.clone(), x.clone(), t.clone(), eps)

        dx = x[0,1] - x[0, 0]
        dt = t[1,0] - t[0, 0]

        return u, dx, dt
    


class PDEDataModule(pl.LightningDataModule):
    def __init__(self, pde_name, data_dir, 
                 batch_size=1, num_workers=1,
                 epsilons = [],
                 n_splits = [160, 20, 20],
                 persistent_workers = False,
                ):
        super().__init__()
        self.pde_name = pde_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.splits = ['train', 'val', 'test']
        
        self.epsilons = epsilons
        self.n_splits = {'train': n_splits[0], 'val': n_splits[1], 'test': n_splits[2]}
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        self.dataset = { 
            split : 
                PDEDataset(
                    pde_name=self.pde_name, 
                    data_dir=self.data_dir, 
                    split=split,
                    epsilons = self.epsilons if split == 'train' else None,
                    n_samples    = self.n_splits[split],
                ) 
            for split in self.splits }
        self.collate_fn = self.custom_collate

    def custom_collate(self, batch):
        us, dxs, dts = zip(*batch)
        return torch.stack(us), torch.stack(dxs), torch.stack(dts)

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