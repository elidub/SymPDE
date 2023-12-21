from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
import os
import pytorch_lightning as pl
from torch import nn

import numpy as np
import torch

from PIL import Image

class MnistDataset(Dataset):

    def __init__(self, mode: str, dim: int = 28, augment: str = 'none', augment_kwargs: dict = {}, N: int = -1, data_dir = '../data/mnist'):
        assert mode in ['train', 'test', 'val']
        assert dim == 28

        self.dim = dim
        self.augment = augment
        self.augment_kwargs = augment_kwargs


        images = np.load(os.path.join(data_dir, mode, f'images.npy'))
        labels = np.load(os.path.join(data_dir, mode, f'labels.npy'))

        N = images.shape[0] if N == -1 else N

        self.images = torch.from_numpy(images[:N])
        self.labels = torch.from_numpy(labels[:N])
    
    def __getitem__(self, index):
        label = self.labels[index]

        if self.augment == 'none':
            image = self.images[index]
        elif self.augment == 'space_translation':
            image = self.space_translation(self.images[index], **self.augment_kwargs)

        return image, label
    
    def space_translation(self, img):
        shift = int(torch.randint(0, self.dim, (1,)))
        img = torch.roll(img, shifts=shift, dims=1)
        return img

    def __len__(self):
        return len(self.labels)
    
class MnistDatasetRotate(Dataset):

    def __init__(self, mode: str, dim: int = 28, augment: str = 'none', augment_kwargs: dict = {}, N: int = -1, data_dir = '../data/mnist'):
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

        dim_org = 28
        images = data[:, :-1].reshape(-1, dim_org, dim_org).astype(np.float32)

        # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
        # we upsample an image by a factor of 3, rotate it and finally downsample it again
        self.resize_large = Resize(87) # to upsample
        self.resize_dim = Resize(dim) # to downsample

        self.dim = dim
        self.augment = augment
        self.augment_kwargs = augment_kwargs

        N = images.shape[0] if N == -1 else N

        self.images_org  = [[]] * N
        self.images_none = torch.empty((N, 1, dim, dim))

        for i in tqdm(range(N), leave=False):
            self.images_org[i] = img = Image.fromarray(images[i], mode='F')
            self.images_none[i] = TF.to_tensor(self.resize_dim(img)).reshape(1, dim, dim)

        self.labels = data[:, -1].astype(np.int64)

        self.images_none = self.images_none[:N]
        self.labels = self.labels[:N]

    def rotate(self, img, r = 45.):
        img = TF.to_tensor(self.resize_dim(self.resize_large(img).rotate(r, Image.BILINEAR))).reshape(1, self.dim, self.dim)
        return img
    
    def __getitem__(self, index):

        label = self.labels[index]

        if self.augment == 'none':
            image = self.images_none[index]
        elif self.augment == 'rotate':
            image = self.rotate(self.images_org[index], **self.augment_kwargs)

        image = image.squeeze(0)

        return image, label

    def __len__(self):
        return len(self.labels)
    

class MnistDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dim,
            # augment,
            # augment_kwargs,
            data_dir, 
            batch_size=1, num_workers=1,
            n_splits = [-1, -1, -1],
            persistent_workers = False,
        ):
        super().__init__()
        self.dim = dim
        # self.augment = augment
        # self.augment_kwargs = augment_kwargs
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
                    dim = self.dim,
                    augment = 'none',
                    augment_kwargs = {},
                    N    = self.n_splits[split],
                    data_dir=self.data_dir, 
                ) 
            for split in self.splits }
        
        self.dataset_aug = { 
            split : 
                MnistDataset(
                    mode = split,
                    dim = self.dim,
                    augment = 'space_translation',
                    augment_kwargs = {},
                    N    = self.n_splits[split],
                    data_dir=self.data_dir, 
                ) 
            for split in self.splits }

        self.collate_fn = self.custom_collate

    def custom_collate(self, batch):
        image, label = zip(*batch)
        return torch.stack(image), torch.stack(label)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.dataset['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
        loader_aug = torch.utils.data.DataLoader(self.dataset_aug['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
        # return loader
        return [loader, loader_aug]
    
    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
        loader_aug = torch.utils.data.DataLoader(self.dataset_aug['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
        # return loader
        return [loader, loader_aug]