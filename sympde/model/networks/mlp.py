import torch
import torchvision

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