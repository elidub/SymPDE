import torch
import torchvision
from torch import nn
from typing import List, Union

from model.networks.linear import LinearP

class MLP(torch.nn.Module):
    def __init__(self, 
            space_length: int,
            bias: bool,
            activation = torch.nn.ReLU,
            linearmodules: List[Union[LinearP, nn.Linear]] = [LinearP, nn.Linear],
            P_init: Union[torch.Tensor, str] = 'none',
            train_weights = True, 
            train_P = False,
        ):
        super().__init__()


        self.mlp = torch.nn.Sequential(
            linearmodules[0](
                in_features=space_length, out_features=space_length, bias=bias, 
                P_init = P_init, train_weights=train_weights, train_P=train_P
            ),
            activation(),
            linearmodules[1](in_features=space_length, out_features=1, bias = bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x