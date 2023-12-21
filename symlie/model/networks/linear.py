import torch
import torchvision
from torch import nn
import math
    
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, transform_type = 'none'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.set_bias = bias

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        self.transform_funcs = {
            'none': self.get_w_index_none,
            'space_translation': self.get_w_index_space_translation,
        }

        assert self.out_features == self.in_features
        self.w_index = self.transform_funcs[transform_type](size = self.out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        From torch.nn.Linear (https://github.com/pytorch/pytorch/blob/af7dc23124a6e3e7b8af0637e3b027f3a8b3fb76/torch/nn/modules/linear.py#L103)
        StackOverflow discussion (https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch#:~:text=To%20initialize%20layers,do%20it%20afterwards)
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_w_index_space_translation(self, size):
        indices1 = torch.arange(-1, size-1)
        indices2 = torch.arange(0, size)
        row_indices = torch.remainder((indices1.view(-1, 1) - indices2), size)
        return size - row_indices -1
    
    def get_w_index_none(self, size):
        return torch.arange(size**2).reshape(size, size)

    def forward(self, x):
        self.weight.data = self.weight.flatten()[self.w_index.flatten()].reshape(self.w_index.shape)

        if self.set_bias:
            out =  x@self.weight.T + self.bias
        else:
            out = x@self.weight.T
        return out