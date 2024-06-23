
import torch
import torchvision
from torch import nn
from typing import List, Union
import torch.nn.functional as F
import math
from torch import nn
import os
import numpy as np
    
class ImplicitLayer(nn.Module):
    def __init__(self, implicit_layer_dim, in_features, out_features, forward_type, pretrained = False):
        super().__init__()

        self.n_features = in_features * out_features + out_features # dim of w and b
        self.in_features = in_features
        self.out_features = out_features


        if implicit_layer_dim == [0]:
            self.implicit_layer = nn.Identity() 
        else:
            self.implicit_layer = torchvision.ops.MLP(
                in_channels=implicit_layer_dim[0], 
                hidden_channels=implicit_layer_dim[1:]
            )

        forward_type_dict = {
            'train': self.forward_train,
        }
        self.forward = forward_type_dict[forward_type]

    def forward_train(self, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if type(b) == nn.Parameter or type(b) == torch.Tensor:
            wb = torch.cat([w, b.view(-1, 1)], dim=1).flatten()

            wb = self.implicit_layer(wb)
            wb = wb.view(self.out_features, self.in_features+1)
            w, b = wb[:, :-1], wb[:, -1]

        elif b is None:
            assert b == None
            w = w.flatten()
            w = self.implicit_layer(w)
            w = w.view(self.out_features, self.in_features) 

        return w, b
    
class CombiMLP(torch.nn.Module):
    # def __init__(self, 
    #         implicit_layer_dims: List[List[int]],
    #         vanilla_layer_dims: List[int],
    #         bias: bool,
    #         activation = torch.nn.ReLU,
    #         pretrained = False,
    #         forward_type = None,
    #         # activation = torch.nn.SiLU,
    #     ):
    def __init__(self, 
            implicit_layer_dims: List[int],
            time_history: int,
            time_future: int,
            space_length: int,
            embed_spacetime: bool,
            hidden_channels: list,
            activation = torch.nn.ReLU,
        ):

        super().__init__()

        self.embed_spacetime = embed_spacetime
        assert self.embed_spacetime == False, 'Not implemented'
        spacetime_dims = 2 if self.embed_spacetime else 0
        self.space_length = space_length

        self.time_history = time_history
        self.time_future = time_future

        self.in_features = (time_history + spacetime_dims)*self.space_length
        self.out_features = time_future*self.space_length

        
        # assert bias == False, 'Not implemented'
        bias=True
        pretrained = False
        forward_type = 'train' 
        assert forward_type is not None

        vanilla_layer_dims = [self.in_features] + hidden_channels + [self.out_features]
        # implicit_layer_dims = [0] * (len(vanilla_layer_dims) - 1)


        implicit_layer_dims = self.convert_implicit_layer_dims(implicit_layer_dims, vanilla_layer_dims, bias)


        assert len(implicit_layer_dims) == len(vanilla_layer_dims)-1, f"len(implicit_layer_dims): {len(implicit_layer_dims)}, len(vanilla_layer_dims): {len(vanilla_layer_dims)}, implicit_layer_dims: {implicit_layer_dims}, vanilla_layer_dims: {vanilla_layer_dims}"

        self.activation = activation()

        self.implicit_layers = nn.ModuleList()
        self.vanilla_layers = nn.ModuleList()
        
        for layer_idx, (implicit_layer_dim, vanilla_layer_dim) in enumerate(zip(implicit_layer_dims, vanilla_layer_dims)):
            in_features, out_features = vanilla_layer_dim, vanilla_layer_dims[layer_idx+1]


            implicit_layer = ImplicitLayer(implicit_layer_dim, in_features, out_features, forward_type=forward_type, pretrained=pretrained)

            self.implicit_layers.append(implicit_layer)

            vanilla_layer = nn.Linear(in_features, out_features, bias=bias)
            self.vanilla_layers.append(vanilla_layer)

        self.reset_parameters_vanilla()

    def convert_implicit_layer_dims(self, implicit_layer_dims, vanilla_layer_dims, bias):
        if np.array(implicit_layer_dims).sum() == 0:
            return np.array(implicit_layer_dims).reshape(-1, 1).tolist()

        vanilla_layer_dims = np.array(vanilla_layer_dims)

        implicit_layer_dims2 = vanilla_layer_dims[:-1] * vanilla_layer_dims[1:]
        if bias:
            implicit_layer_dims2 += vanilla_layer_dims[1:]
        implicit_layer_dims2 = implicit_layer_dims2#.reshape(-1, 1).tolist()

        return [[implicit_layer_dim2] * implicit_layer_dim for implicit_layer_dim, implicit_layer_dim2 in zip(implicit_layer_dims, implicit_layer_dims2)]


    def reset_parameters_vanilla(self):
        for layer in self.vanilla_layers:
            layer.reset_parameters()

    def mlp(self, x: torch.Tensor) -> List[torch.Tensor]:

        self.x_ins = []

        for layer_idx, (implicit_layer, vanilla_layer) in enumerate(zip(self.implicit_layers, self.vanilla_layers)):

            with torch.no_grad():
                self.x_ins.append(x)

            weight, bias = implicit_layer(vanilla_layer.weight, vanilla_layer.bias)
            x = F.linear(x, weight, bias)

            if layer_idx != len(self.vanilla_layers)-1:
                x = self.activation(x)


        return x

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:

        batch_size, nx, nt = u.shape
        assert nx == self.space_length
        assert nt == self.time_history 

        nx = u.shape[1] 
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
                       dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1) if self.embed_spacetime else u

        x = x.reshape(batch_size, self.in_features)
        x = self.mlp(x)
        x = x.reshape(batch_size, self.space_length, self.time_future)

        return x