import torch
import torchvision
from torch import nn
from typing import List, Union

from model.networks.linear import LinearP

class MLP(torch.nn.Module):
    def __init__(self, 
            in_features: int,
            bias: bool,
            device: str,
            activation = torch.nn.ReLU,
            linearmodules: List[Union[LinearP, nn.Linear]] = [LinearP, nn.Linear],
            out_features: int = 1,
            P_init: Union[torch.Tensor, str] = 'none',
            train_weights = True, 
            train_P = False,
        ):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            linearmodules[0](
                in_features=in_features, out_features=in_features, bias=bias, device=device,
                P_init = P_init, train_weights=train_weights, train_P=train_P
            ),
            activation(),
            linearmodules[1](in_features=in_features, out_features=out_features, bias = bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x