import torch
import emlp.nn.pytorch as nn
from emlp.groups import SO13
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from emlp.datasets import ParticleInteraction

trainset = ParticleInteraction(300) # Initialize dataset with 1000 examples
testset = ParticleInteraction(1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size=500
lr=3e-3
max_epochs=500

group = SO13()
num_layers = 3
ch = 3

net = nn.MLP
net = nn.EMLP

model = net(trainset.rep_in,trainset.rep_out,group=group,num_layers=num_layers,ch=ch).to(device)

x = torch.randn(batch_size,trainset[0][0].size).to(device) # generate some random data
print(x.dtype)

model(x)

print(model)