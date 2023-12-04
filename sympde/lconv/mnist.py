from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from tqdm.auto import tqdm
import os
import pytorch_lightning as pl
from torch import nn

import numpy as np
import torch

from PIL import Image

class MnistDataset(Dataset):

    def __init__(self, mode: str, r: float, digit: int = None, dim: int = 28, N: int = -1, data_dir = '../data/mnist'):
        assert mode in ['train', 'test', 'val']

        if mode == "train":
            file = os.path.join(data_dir, "mnist_train.amat")
        else:
            file = os.path.join(data_dir, "mnist_test.amat")

        data = np.loadtxt(file)

        if mode == 'val':
            data = data[:25000]
        elif mode == 'test':
            data = data[25000:]


        images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)

        # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
        # we upsample an image by a factor of 3, rotate it and finally downsample it again
        resize1 = Resize(87) # to upsample
        resize2 = Resize(dim) # to downsample

        totensor = ToTensor()

        self.images = torch.empty((images.shape[0], 1, dim, dim))
        self.images_rot = torch.empty((images.shape[0], 1, dim, dim))
        N = images.shape[0] if N == -1 else N
        for i in tqdm(range(N), leave=False):
            img = images[i]
            img = Image.fromarray(img, mode='F')
            self.images[i] = totensor(resize2(img)).reshape(1, dim, dim)
            self.images_rot[i] = totensor(resize2(resize1(img).rotate(r, Image.BILINEAR))).reshape(1, dim, dim)

        self.labels = data[:, -1].astype(np.int64)

        if digit is not None:
            idxs = np.where(self.labels == digit)[0]
            self.images = self.images[idxs]
            self.labels = self.labels[idxs]
            self.images_rot = self.images_rot[idxs]



    def __getitem__(self, index):
        image, label, image_rot = self.images[index], self.labels[index], self.images_rot[index]

        # return image, label
        return image, image_rot

    def __len__(self):
        return len(self.labels)
    

class Reshape(nn.Module):
    def __init__(self,shape=None):
        self.shape = shape
        super().__init__()
    def forward(self,x):
        return x.view(-1,*self.shape)
    


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, r, dim, digit,
                 data_dir, 
                 batch_size=1, num_workers=1,
                 n_splits = [-1, -1, -1],
                 persistent_workers = False,
                ):
        super().__init__()
        self.r = r
        self.dim = dim
        self.digit = digit
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = ['train', 'val', 'test']
        
        self.n_splits = {'train': n_splits[0], 'val': n_splits[1], 'test': n_splits[2]}
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        self.dataset = { 
            split : 
                MnistDataset(
                    mode = split,
                    r = self.r,
                    dim = self.dim,
                    digit = self.digit,
                    data_dir=self.data_dir, 
                    N    = self.n_splits[split],
                ) 
            for split in self.splits }
        self.collate_fn = self.custom_collate

    def custom_collate(self, batch):
        image, image_rot = zip(*batch)
        return torch.stack(image), torch.stack(image_rot)

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