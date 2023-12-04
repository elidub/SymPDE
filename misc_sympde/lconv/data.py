from time import time

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from numpy import cos, sin, arange, trace, pi, newaxis, sum, eye, array, mgrid, diag
from numpy.linalg import norm, matrix_power

class FixedRot(datasets.VisionDataset):
    num_targets = 10
    def __init__(self,*args,angle =pi/3,N=50000,size=(7,7),
                 train=True,dataseed=0,**kwargs):
        super().__init__(*args,**kwargs)
        if not train: 
            dataseed += 1
            N = int(0.2*N)
        torch.manual_seed(dataseed)
        angles = torch.ones(N)*angle # torch.rand(N)*2*np.pi
        self.data = torch.rand(N,1,*size)-.5
        print(N, self.data.shape)
        with torch.no_grad():
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            # affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
            # affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
            
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data_rot = F.grid_sample(self.data, flowgrid)
    def __getitem__(self,idx):
        return self.data[idx], self.data_rot[idx]
    
    def __len__(self):
        return len(self.data)
    
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation


class Reshape(nn.Module):
    def __init__(self,shape=None):
        self.shape = shape
        super().__init__()
    def forward(self,x):
        return x.view(-1,*self.shape)
    

