import torch
import torchvision

from model.networks.utils import PrintLayer

class MLP(torch.nn.Module):
    def __init__(self, 
            time_history: int,
            time_future: int,
            hidden_channels: list,
            activation = torch.nn.ReLU,
        ):
        # super(MLP, self).__init__()
        super().__init__()
        self.time_history = time_history
        self.time_future = time_future

        channels = hidden_channels + [time_future]
        self.mlp = torchvision.ops.MLP(
            in_channels = time_history + 2, # +2 for dx and dt 
            hidden_channels = channels,
            norm_layer = None,
            activation_layer = activation,
        )

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:

        nx = u.shape[1] 
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
                       dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1)
        x = self.mlp(x)
        return x
    
class CustomMLP(torch.nn.Module):
    def __init__(self, 
            time_history: int,
            time_future: int,
            hidden_channels: list,
            activation = torch.nn.ReLU,
            embed_spacetime = False,
            equiv = 'none',
        ):
        super().__init__()

        self.embed_spacetime  = embed_spacetime
        spacetime_dims = 2 if self.embed_spacetime else 0

        self.time_history = time_history + spacetime_dims
        self.time_future = time_future

        layers = []
        layer_sizes = [time_history] + hidden_channels# + [time_future]

        for layer_index in range(1, len(layer_sizes)):
            layers += [ torch.nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                        activation() ]
        layers += [ torch.nn.Linear(layer_sizes[-1], time_future) ]

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
            
        nx = u.shape[1] 
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
                    dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1) if self.embed_spacetime else u

        x = self.mlp(x)
        return x
