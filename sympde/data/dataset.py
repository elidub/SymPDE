import os
import torch
import numpy as np
import pytorch_lightning as pl

from data.utils import load_obj

class PDEDataset(torch.utils.data.Dataset):
    def __init__(
        self, pde_name, data_dir, split="val", transform=None, target_transform=None
    ):
        self.split = split
        split_dir = os.path.join(data_dir, split)

        self.transform = transform
        self.target_transform = target_transform

        self.us, self.dx, self.dt = load_obj(os.path.join(split_dir, f"{pde_name}"))

    def __len__(self):
        return len(self.us)
    
    def __getitem__(self, idx):

        u = self.us[idx]

        if self.transform:
            u = self.transform(u)

        if self.target_transform:
            u = self.target_transform(u)

        u = torch.from_numpy(u)

        # return u.float(), self.dx, self.dt
        return u.float(), torch.tensor(self.dx, dtype = torch.float32), torch.tensor(self.dt, dtype = torch.float32)

class PDEDataModule(pl.LightningDataModule):
    def __init__(self, pde_name, data_dir, batch_size=1, num_workers=1):
        self.pde_name = pde_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.splits = ['train', 'val', 'test']

    def setup(self, stage=None):
        self.dataset = { split : PDEDataset(pde_name=self.pde_name, data_dir=self.data_dir, split=split) for split in self.splits }
        self.collate_fn = self.custom_collate

    def custom_collate(self, batch):
        us, dx, dt = zip(*batch)
        # return torch.stack(us), torch.stack(dx).float(), torch.stack(dt).float()
        return torch.stack(us), torch.stack(dx), torch.stack(dt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['validation'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)