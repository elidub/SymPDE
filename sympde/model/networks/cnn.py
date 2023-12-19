"""
Adapted from Brandstetter, J., Welling, M., Worrall, D.E., 2022. Lie Point Symmetry Data Augmentation for Neural PDE Solvers. https://doi.org/10.48550/arXiv.2202.07643
    https://github.com/brandstetter-johannes/LPSDA/blob/master/experiments/models_cnn.py
"""

import sys
import numpy as np
import torch
from collections import OrderedDict
from typing import Tuple
from torch import nn
from torch.nn import functional as F
import logging

from model.networks.single_sym.magnitude import Conv1dMag, Conv2dMag

class CNN(nn.Module):
    '''
    A simple baseline 1D Res CNN approach, the time dimension is stacked in the channels
    '''
    def __init__(self,
                 time_history: int,
                 time_future: int,
                 embed_spacetime: bool,
                 width: int = 128,
                 padding_mode: str = f'circular'):
        """
        Initialize the simple CNN architecture. It contains 8 1D CNN-layers with skip connections
        and increasing receptive field.
        The input to the forward pass has the shape [batch, time_history, x].
        The output has the shape [batch, time_future, x].
        Args:
            time_history (int): input timesteps of the trajectory
            time_future (int): output timesteps of the trajectory
            width (int): hidden channel dimension
            padding_mode (str): circular mode as default for periodic boundary problems
        """
        super().__init__()
        self.time_history = time_history
        self.time_future = time_future
        self.embed_spacetime = embed_spacetime
        self.width = width
        self.padding_mode = padding_mode

        spacetime_dims = 2 if self.embed_spacetime else 0
        self.fc0 = nn.Linear(self.time_history + spacetime_dims, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

        kernel_sizes = [3, 5, 5, 7, 7, 9, 9, 15]

        conv_layers = []
        for kernel_size in kernel_sizes:
            conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=kernel_size,
                                         padding=(kernel_size - 1) // 2, padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=3, padding=1,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=5, padding=2,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=5, padding=2,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=7, padding=3,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=7, padding=3,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=9, padding=4,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=9, padding=4,
        #                              padding_mode=self.padding_mode, bias=True))
        # conv_layers.append(nn.Conv1d(in_channels=self.width, out_channels=self.width, kernel_size=15, padding=7,
        #                              padding_mode=self.padding_mode, bias=True))

        self.conv_layers = nn.ModuleList(conv_layers)
        for layer in self.conv_layers:
            nn.init.xavier_uniform_(layer.weight)

    def __repr__(self):
        return f'CNN'

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, time_history, x].
        1. Add dx and dt as channel dimension to the time_history, repeat for every x
        2. Lift the input to the desired channel dimension by self.fc0
        3. 8 CNN layers with increasing receptive field and skip connections
        4. Project from the channel space to the output space by self.fc1 and self.fc2.
        The output has the shape [batch, time_future, x].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x]
            dx (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns:
            torch.Tensor: output has the shape [batch, time_future, x]
        """

        # ADAPTION: use nx = u.shape[1] instead of nx = self.pde.nx
        nx = u.shape[1] 

        # todo: rewrite training method and forward pass without permutation
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
                    dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1) if self.embed_spacetime else u
        logging.debug(f'embed spacetime: {x.shape}')
        x = self.fc0(x) # batch, space, time_history
        logging.debug(f'fc0: {x.shape}')
        x = x.permute(0, 2, 1) # -> batch, c, space
        logging.debug(f'permute: {x.shape}')

        for layer in self.conv_layers:
            x = F.elu(x + layer(x))
        logging.debug(f'conv_layers: {x.shape}')

        x = x.permute(0, 2, 1) # -> batch, space, c
        logging.debug(f'permute: {x.shape}')
        x = self.fc1(x) # batch, space, c
        logging.debug(f'fc1: {x.shape}')
        x = F.elu(x)
        x = self.fc2(x) # batch, space, time_future
        logging.debug(f'fc2: {x.shape}')
        return x


def get_nn_modules(equiv: str) -> Tuple[nn.Module, nn.Module]:
        if equiv == 'mag':
            Conv1d = Conv1dMag
            # BatchNorm1d = nn.Identity
            # Conv1d = nn.Conv1d
            BatchNorm1d = nn.BatchNorm1d
        elif equiv == 'none':
            Conv1d = nn.Conv1d
            BatchNorm1d = nn.BatchNorm1d
        else:
            raise NotImplementedError(f'Unknown equiv {equiv}')
        return Conv1d, BatchNorm1d
        

class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, equiv: str, stride: int=1):
        """
        Initializes the 1D basic ResNet block.
        Args:
            in_planes (int): input channels to the 1D basic ResNet block
            planes (int): output channels of the 1D basic ResNet block
            stride (int): stride used for filters
        """
        # super(BasicBlock1d, self).__init__()
        super().__init__()

        self.Conv1d, self.BatchNorm1d = get_nn_modules(equiv)

        self.relu = F.relu

        self.conv1 = self.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = self.BatchNorm1d(out_channels)
        self.conv2 = self.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = self.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            print('shortcut')
            self.shortcut = nn.Sequential(
                self.Conv1d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                self.BatchNorm1d(self.expansion*out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 1D basic ResNet block.
        2 conv layers with batch norm and skip connection
        Args:
            x (torch.Tensor): input tensor of shape [batch, in_channel, x]
        Returns:
            torch.Tensor: output of shape [batch, out_channel, x]
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

class ResNet_conv(nn.Module):
    def __init__(self,
                block: nn.Module,
                num_blocks: list,
                time_history: int,
                time_future: int,
                embed_spacetime: bool,
                equiv: str,
                width: int = 256):
        """
        Initialize the 1D ResNet architecture. It contains 4 1D basic blocks.
        In contrast to standard ResNet architectues, the spatial dimension is not decreasing.
        The input to the forward pass has the shape [batch, time_history, x].
        The output has the shape [batch, time_future, x].
        Args:
            block (nn.Module): basic block for ResNet
            num_blocks (list): number of layers used in each basic block
            time_history (int): input timesteps of the trajectory
            time_future (int): output timesteps of the trajectory
            width (int): hidden channel dimension
        """
        # super(ResNet, self).__init__()
        super().__init__()
        self.in_channels = width
        self.time_history = time_history
        self.time_future = time_future
        self.embed_spacetime = embed_spacetime
        self.equiv = equiv

        self.Conv1d, self.BatchNorm1d = get_nn_modules(equiv)

        spacetime_dims = 2 if self.embed_spacetime else 0
        self.conv1 = self.Conv1d(time_history+spacetime_dims, width, kernel_size=1, bias=True)

        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, width, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, width, num_blocks[3], stride=1)

        # self.fc1 = nn.Linear(width, time_future)
        self.conv_out = self.Conv1d(width, time_future, kernel_size=1, bias=True)

    def _make_layer(self, block: nn.Module, channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Building basic ResNet layers out for basic 1D building blocks.
        Args:
             block (nn.Module): Basic 1D building blocks
             channels (int): output channels
             num_blocks (int): how many building blocks does one ResNet layer contain
             stride (int): stride
        Returns:
            nn.Sequential: the basic ResNet layer
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, self.equiv, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return f'ResNet'

    def forward(self, u: torch.Tensor, dx: torch.Tensor = 0, dt: torch.Tensor = 0) -> torch.Tensor:
        """
        Forward pass of a 1D ResNet.
        The input to the forward pass has the shape [batch, time_history, x].
        1. Add dx and dt as channel dimension to the time_history, repeat for every x
        2. Lift the input to the desired channel dimension by self.fc0
        3. 4 basic ResNet layers with flexible number of basic building blocks
        4. Project from the channel space to the output space by self.fc1.
        The output has the shape [batch, time_future, x].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x]
            dx (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns:
            torch.Tensor: output has the shape [batch, time_future, x]
        """

        # ADAPTION: use nx = u.shape[1] instead of nx = self.pde.nx
        nx = u.shape[1] 

        # [b, x, c] = [b, x, t+2]
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
            dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1) if self.embed_spacetime else u

        # [b, x, c] -> [b, c, x]
        # x = x.permute(0, 2, 1)
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.gelu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv_out(x)
        # x = x.permute(0, 2, 1)
        # x = self.fc1(x)
        return x
    


class ResNet(nn.Module):
    def __init__(self,
                block: nn.Module,
                num_blocks: list,
                time_history: int,
                time_future: int,
                embed_spacetime: bool,
                width: int = 256):
        """
        Initialize the 1D ResNet architecture. It contains 4 1D basic blocks.
        In contrast to standard ResNet architectues, the spatial dimension is not decreasing.
        The input to the forward pass has the shape [batch, time_history, x].
        The output has the shape [batch, time_future, x].
        Args:
            block (nn.Module): basic block for ResNet
            num_blocks (list): number of layers used in each basic block
            time_history (int): input timesteps of the trajectory
            time_future (int): output timesteps of the trajectory
            width (int): hidden channel dimension
        """
        # super(ResNet, self).__init__()
        super().__init__()
        self.in_channels = width
        self.time_history = time_history
        self.time_future = time_future
        self.embed_spacetime = embed_spacetime

        spacetime_dims = 2 if self.embed_spacetime else 0
        self.conv1 = nn.Conv1d(time_history+spacetime_dims, width, kernel_size=1, bias=True)

        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, width, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, width, num_blocks[3], stride=1)
        self.fc1 = nn.Linear(width, time_future)

    def _make_layer(self, block: nn.Module, channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Building basic ResNet layers out for basic 1D building blocks.
        Args:
             block (nn.Module): Basic 1D building blocks
             channels (int): output channels
             num_blocks (int): how many building blocks does one ResNet layer contain
             stride (int): stride
        Returns:
            nn.Sequential: the basic ResNet layer
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, 'none', stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return f'ResNet'

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a 1D ResNet.
        The input to the forward pass has the shape [batch, time_history, x].
        1. Add dx and dt as channel dimension to the time_history, repeat for every x
        2. Lift the input to the desired channel dimension by self.fc0
        3. 4 basic ResNet layers with flexible number of basic building blocks
        4. Project from the channel space to the output space by self.fc1.
        The output has the shape [batch, time_future, x].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x]
            dx (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns:
            torch.Tensor: output has the shape [batch, time_future, x]
        """

        # ADAPTION: use nx = u.shape[1] instead of nx = self.pde.nx
        nx = u.shape[1] 

        # [b, x, c] = [b, x, t+2]
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
            dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1) if self.embed_spacetime else u

        # [b, x, c] -> [b, c, x]
        x = x.permute(0, 2, 1)
        # print(1, x.shape)
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.gelu(self.conv1(x))
        # print(2, x.shape)
        x = self.layer1(x)
        # print(3, x.shape)
        x = self.layer2(x)
        # print(4, x.shape)
        x = self.layer3(x) # batch, time_channel, space
        # print(5, x.shape)
        x = self.layer4(x) # batch, time_channel, space
        # print(4, x.shape)

        x = x.permute(0, 2, 1)
        # print('permute', x.shape)
        x = self.fc1(x) # batch, space, time_future
        # print('fc1', x.shape)
        return x