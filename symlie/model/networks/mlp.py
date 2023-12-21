import torch
import torchvision
from torch import nn
from typing import List, Union

from model.networks.linear import MyLinear

class MLP(torch.nn.Module):
    def __init__(self, 
            transform_type: str,
            space_length: int,
            linearmodules: List[Union[MyLinear, nn.Linear]],
            bias: bool,
            activation = torch.nn.ReLU,
        ):
        super().__init__()

        # assert linearmodule in [nn.Linear, MyLinear]
        assert len(linearmodules) == 2, linearmodules
        assert linearmodules == [MyLinear, nn.Linear], linearmodules

        self.mlp = torch.nn.Sequential(
            nn.Flatten(),
            linearmodules[0](in_features=space_length**2, out_features=space_length**2, bias=bias, transform_type=transform_type),
            activation(),
            linearmodules[1](in_features=space_length**2, out_features=10, bias = bias),
            nn.Softmax(dim=1),
        )

        # self.mlp = torch.nn.Sequential(
        #     nn.Flatten(),
        #     linearmodules[0](in_features=space_length**2, out_features=space_length, bias = True),
        #     activation(),
        #     linearmodules[1](in_features=space_length, out_features=10, bias = True),
        #     nn.Softmax(dim=1),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
    

class MLPTorch(torch.nn.Module):
    def __init__(self, 
            space_length: int,
            activation = torch.nn.ReLU,
        ):
        super().__init__()

        self.flatten = nn.Flatten()
        self.mlp = torchvision.ops.MLP(
            in_channels = space_length**2,  
            hidden_channels = [space_length**2, 10],
            activation_layer = activation,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.mlp(x)
        x = self.softmax(x)
        return x