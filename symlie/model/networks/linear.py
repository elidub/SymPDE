import torch
from torch import nn
from typing import List, Union


class CalculatedP:
    def __init__(self, size, device = 'cpu'):
        self.size = size
        self.device = device

        self.transform_funcs = {
            'none': self.get_none,
            'randn': self.get_randn,
            'space_translation': self.get_space_translation,
            'kernelconv': self.get_kernelconv,
        }

    def get_none(self):
        out = torch.eye(self.size**2).reshape(self.size**2, self.size**2)
        return out
    
    def get_randn(self):
        w_index = torch.randn(self.size**2, self.size**2)
        return w_index
    
    def get_space_translation(self):
        w1 = torch.zeros(self.size)
        w1[0] = 1.
        w2 = torch.stack([torch.roll(w1, shifts = shift) for shift in range(self.size)])
        w3 = torch.cat([torch.roll(w2, shifts = (shift, 0), dims = (0,1)) for shift in range(self.size)])

        w_index = torch.zeros(self.size**2, self.size**2)
        w_index[:, :self.size] = w3
        return w_index
    
    def get_convolution(self):
        raise NotImplementedError
    
    def get_kernelconv(self):
        kernel_size = 3
        size = self.size

        p1 = torch.zeros(size, size)
        p1[:kernel_size, :kernel_size] = torch.eye(kernel_size)

        p2s = torch.vstack([torch.roll(p1, shifts = (shift-1,0), dims = (0,1)) for shift in range(size)])

        p = torch.zeros(size**2, size**2)
        p[:, :size] = p2s

        # w_index = torch.zeros(self.size**2, self.size**2)
        # for s in range(self.size):
            # w_index[s*self.size+s, s*self.size+s] = 1

        return p
    

class LinearP(nn.Module, CalculatedP):
    def __init__( 
            self, in_features, out_features, bias, 
            device = 'cpu',
            P_init: Union[torch.Tensor, str] = 'none',
            train_weights = True, train_P = False,
            svd_rank: int = None,
        ):
        nn.Module.__init__(self)
        CalculatedP.__init__(self, size = out_features, device = device)


        self.in_features = in_features
        self.out_features = out_features
        assert self.out_features == self.in_features # Not implemented yet
        assert bias == False # Not implemented yet
        assert train_weights is not train_P
        if type(P_init) == str:
            assert P_init in ['none', 'randn', 'space_translation', 'convolution', 'kernelconv']
        self.set_bias = bias
        self.train_weights = train_weights
        self.train_P = train_P
        self.svd = True if svd_rank is not None else False
        assert train_weights != train_P

        # Initalize P
        if self.svd:
            assert P_init == 'randn' and train_P
            self.U = torch.randn(self.out_features**2, svd_rank, device = self.device)
            self.S = torch.randn(svd_rank, device = self.device)
            self.V = torch.randn(svd_rank, self.out_features**2, device = self.device)
        elif type(P_init) == str:
            self.P = self.transform_funcs[P_init]().to(self.device)
        else:
            self.P = P_init

        # if not self.train_P:
        #     self.reset_parameters(batch_size=None)

        if train_weights:
            self.reset_parameters(batch_size=None)
            assert P_init not in ['randn']
            self.weight = nn.Parameter(self.weight)
            # self.bias = nn.Parameter(self.bias)
        else:
            if self.svd:
                self.U = nn.Parameter(self.U)
                self.S = nn.Parameter(self.S)
                self.V = nn.Parameter(self.V)
            else:
                self.P = nn.Parameter(self.P)

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

    @staticmethod
    def normalize_P(P):
        # P = torch.abs(P)
        # P = torch.exp(P)
        P = P / torch.linalg.norm(P, ord = 1, dim = 1).reshape(-1, 1)

        # P_sum = torch.sum(P, dim = 1)
        # assert torch.allclose(P_sum, torch.ones_like(P_sum), atol=1e-5, rtol=1e-4), P_sum-torch.ones_like(P_sum)
        return P

    def forward(self, x, batch_size = None, P = None, normalize_P = True):

        if P is None:
            if self.svd:
                P = self.U @ torch.diag(self.S) @ self.V
            else:
                P = self.P

        if normalize_P:
            P = self.normalize_P(P)

        if (not self.train_P) or (batch_size is None):
            weight = (P @ self.weight.flatten()).reshape(self.weight.shape)
            out = x @ weight.T
        else:
            weight = (P @ self.weight.flatten(1).T).T.reshape(self.weight.shape)
            out = torch.einsum('bi,boi->bo', x, weight)

        if self.set_bias: out = out + self.bias
        return out