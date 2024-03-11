import torch
from torch import nn
import torchvision
from typing import List, Union


class LinearImplicit(nn.Module):
    def __init__( 
            self, in_features, out_features, bias, 
            hidden_implicit_layers: List[int],
            device = 'cpu',
            # P_init: Union[torch.Tensor, str] = 'none',
            train_weights = True, train_P = False,
            # svd_rank: int = None,
        ):
        nn.Module.__init__(self)
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        assert self.out_features == self.in_features # Not implemented yet
        assert bias == False # Not implemented yet
        assert train_weights is not train_P
        self.set_bias = bias
        self.train_weights = train_weights
        self.train_P = train_P
        assert train_weights != train_P

        # Initalize P
        self.implicit_P = self.setup_implict_P(hidden_implicit_layers = hidden_implicit_layers)


        if not self.train_P or True:
            self.reset_parameters(batch_size=None)

        if train_weights:
            self.weight = nn.Parameter(self.weight)

            # Turn off gradients for implicit_P
            for implicit_P_param in self.implicit_P.parameters():
                implicit_P_param.requires_grad = False

    def setup_implict_P(self, hidden_implicit_layers: List[int]):
        implicit_in_features = implicit_out_features = self.in_features*self.out_features
        implicit_layers = [implicit_in_features] + hidden_implicit_layers + [implicit_out_features]
        layers = []
        for i in range(len(implicit_layers)-1):
            layers.append(nn.Linear(implicit_layers[i], implicit_layers[i+1]))
            layers.append(nn.ReLU())
        implicit_P = nn.Sequential(*layers)
        return implicit_P

    def reset_parameters(self, batch_size = None) -> None:
        """
        From torch.nn.Linear (https://github.com/pytorch/pytorch/blob/af7dc23124a6e3e7b8af0637e3b027f3a8b3fb76/torch/nn/modules/linear.py#L103)
        StackOverflow discussion (https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch#:~:text=To%20initialize%20layers,do%20it%20afterwards)
        """
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)

        if batch_size is None:
            self.weight = torch.randn(self.out_features, self.in_features, device = self.device)
            # self.bias   = torch.randn(self.out_features, device = self.device)
        else:
            self.weight = torch.randn(batch_size, self.out_features, self.in_features, device = self.device)
            # self.bias   = torch.randn(batch_size, self.out_features, device = self.device)

    # @staticmethod
    # def normalize_P(P):
    #     # P = torch.abs(P)
    #     # P = torch.exp(P)
    #     P = P / torch.linalg.norm(P, ord = 1, dim = 1).reshape(-1, 1)

    #     # P_sum = torch.sum(P, dim = 1)
    #     # assert torch.allclose(P_sum, torch.ones_like(P_sum), atol=1e-5, rtol=1e-4), P_sum-torch.ones_like(P_sum)
    #     return P

    def forward(self, x, batch_size = None, normalize_P = True):

        # if normalize_P:
            # P = self.normalize_P(P)

        if (not self.train_P) or (batch_size is None):
            weight = self.implicit_P(self.weight.flatten()).reshape(self.weight.shape)
            out = x @ weight.T
        else:
            weight = self.implicit_P(self.weight.flatten(1)).reshape(self.weight.shape)
            out = torch.einsum('bi,boi->bo', x, weight)

        if self.set_bias: out = out + self.bias
        return out