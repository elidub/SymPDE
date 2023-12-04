from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from tqdm.auto import tqdm

import numpy as np
import torch

from PIL import Image

class MnistDataset(Dataset):

    def __init__(self, mode, r: float, dim: int = 28, N: int = None):
        assert mode in ['train', 'test']

        if mode == "train":
            file = "mnist/mnist_train.amat"
        else:
            file = "mnist/mnist_test.amat"

        data = np.loadtxt(file)

        images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)

        # images are padded to have shape 29x29.
        # this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
        pad = Pad((0, 0, 1, 1), fill=0)

        # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
        # we upsample an image by a factor of 3, rotate it and finally downsample it again
        resize1 = Resize(87) # to upsample
        resize2 = Resize(dim) # to downsample

        totensor = ToTensor()

        self.images = torch.empty((images.shape[0], 1, dim, dim))
        self.images_rot = torch.empty((images.shape[0], 1, dim, dim))
        N = images.shape[0] if N is None else N
        for i in tqdm(range(N), leave=False):
            img = images[i]
            img = Image.fromarray(img, mode='F')
            # r = (np.random.rand() * 360.)
            # r = 45.
            self.images[i] = totensor(resize2(img)).reshape(1, dim, dim)
            self.images_rot[i] = totensor(resize2(resize1(img).rotate(r, Image.BILINEAR))).reshape(1, dim, dim)

        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label, image_rot = self.images[index], self.labels[index], self.images_rot[index]

        # return image, label
        return image, image_rot

    def __len__(self):
        return len(self.labels)