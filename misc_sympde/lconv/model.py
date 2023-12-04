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
from numpy.random import randn, rand, seed

from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, legend, show, xlim, ylim, subplot, grid, savefig, imshow, colorbar, tight_layout, text, xticks, yticks

# includes residual
       
class Lconv_core(nn.Module):
    """ L-conv layer with full L """
    def __init__(self,d,num_L=1,cin=1,cout=1,rank=8):
        """
        L:(num_L, d, d)
        Wi: (num_L, cout, cin)
        """
        super().__init__()
        self.L = nn.Parameter(torch.Tensor(num_L, d, d))
        self.Wi = nn.Parameter(torch.Tensor(num_L, cout, cin))
        
        # initialize weights and biases
        nn.init.kaiming_normal_(self.L) 
        nn.init.kaiming_normal_(self.Wi)
                
    def forward(self, x):
        # x:(batch, channel, flat_d)
        # h = (x + Li x Wi) W0
        y = torch.einsum('kdf,bcf,koc->bod', self.L, x, self.Wi ) +x #+ self.b        
        return y

# includes residual

class Lconv_grid(nn.Module):
    """ L-conv using an inferred grid in the data. 
    """
    def __init__(self,idx, k,d,cin,cout):
        super().__init__()
        self.idx = idx
        self.L_sparse = nn.Parameter(torch.Tensor(*idx.shape, k)) 
        self.W = nn.Parameter(torch.Tensor(k, cin, cout))
        self.b = nn.Parameter(torch.Tensor(1, cout, 1))
        # the bias is not complete, but more efficient 
#         self.b = nn.Parameter(torch.Tensor(1,k, cout, 1))
        
        # initialize weights and biases
        nn.init.kaiming_normal_(self.L_sparse, mode='fan_in') 
        nn.init.kaiming_normal_(self.W)      
        nn.init.kaiming_normal_(self.b)
                
    def forward(self, x):
        # x:(batch, channel, flat_d)
        return torch.einsum('bcdi,dik,kco->bod',x[:,:,self.idx], self.L_sparse, self.W ) + self.b