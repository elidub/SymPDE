import torch
import torchvision
from torch import nn
import math
import matplotlib.pyplot as plt

    
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, transform_type = 'none', reset_parameters = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert self.out_features == self.in_features
        self.set_bias = bias
        self.transform_type = transform_type

        # self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # self.bias   = nn.Parameter(torch.randn(out_features))
        self.weight = torch.randn(out_features, in_features)
        self.bias   = torch.randn(out_features)

        self.w_index = NotImplemented
        self.w_index_param = nn.Parameter(torch.randn(out_features*in_features+1,out_features*in_features))

        self.transform_funcs = {
            'none': self.get_w_index_none,
            'space_translation': self.get_w_index_space_translation,
            'convolution': self.get_w_index_convolution,
            'kernelconv': self.get_w_index_kernelconv
        }

        if reset_parameters:
            self.reset_parameters()

        self.w_index = self.transform_funcs[self.transform_type](size = self.out_features)

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
        
        w1 = torch.arange(size)
        dim = int(math.sqrt(size))
        w2 = w1.reshape(dim, dim)

        w_index = torch.cat([torch.stack([torch.roll(w2_i.repeat(dim), shifts = shift, dims = 0) for shift in range(dim)]) for w2_i in w2]).T


        return w_index.T
                
        ## Incorrect (?) implementation
        # indices1 = torch.arange(-1, size-1)
        # indices2 = torch.arange(0, size)
        # row_indices = torch.remainder((indices1.view(-1, 1) - indices2), size)
        # return (size - row_indices -1).T
    
    def get_w_index_none(self, size):
        out = torch.arange(size**2).reshape(size, size)
        return out
    
    def get_w_index_convolution(self, size):

        kernel_size = 3
        assert kernel_size == 3

        # dim = int(math.sqrt(size))
        dim = size
        
        w1 = torch.full((dim,), -1)
        w1[[-1, 0, 1]] = torch.arange(kernel_size)

        w_index = torch.stack([torch.roll(w1, shifts = i, dims = 0) for i in range(dim)])

        return w_index
    
    def get_w_index_kernelconv(self, size):

        w_index = torch.diag(torch.arange(size))

        w_index[~torch.eye(size, dtype=bool)] = -1

        return w_index

    

    def forward(self, x, w_index = None):
        # w_index = self.w_index if w_index is None else w_index
        
        # self.weight.data = self.weight.flatten()[w_index.flatten()].reshape(w_index.shape)
        # if self.set_bias:
        #     out =  x@self.weight.T + self.bias
        # else:
        #     out = x@self.weight.T
        # return out

        indices = torch.max(self.w_index_param, dim = 0).indices
        indices = torch.where(indices < self.out_features*self.in_features, indices, -1)

        # indices = self.w_index.flatten()

        weight = self.weight.flatten()[indices].reshape(self.weight.shape)
        # weight = self.weight.flatten()[w_index.flatten()].reshape(w_index.shape)

        weight = torch.where(indices.reshape(weight.shape) >= 0, weight, 0)

        # print(x.shape)
        # print(x)
        # plt.imshow(weight)
        # plt.show()


        if self.set_bias:
            out =  x@weight.T + self.bias
        else:
            out = x@weight.T
        return out
    


class MyLinearPw(nn.Module):
    def __init__(self, in_features, out_features, bias, transform_type = 'none', reset_parameters = True, w_index_trained = None, train_layer = False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert self.out_features == self.in_features
        assert bias == False
        self.set_bias = bias
        self.transform_type = transform_type

        # self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # self.bias   = nn.Parameter(torch.randn(out_features))
        self.weight = torch.randn(out_features, in_features)
        self.bias   = torch.randn(out_features)

        if train_layer:
            self.weight = nn.Parameter(self.weight)
            self.bias = nn.Parameter(self.bias)
            assert self.transform_type != 'train_Pw'

        self.transform_funcs = {
            'train_Pw': self.get_w_index_train,
            'trained': lambda size: w_index_trained,
            'none': self.get_w_index_none,
            'space_translation': self.get_w_index_space_translation,
            'convolution': self.get_w_index_convolution,
            'kernelconv': self.get_w_index_kernelconv,
        }

        # if reset_parameters:
            # self.reset_parameters()

        self.w_index = self.transform_funcs[self.transform_type](size = self.out_features)

    def reset_parameters(self) -> None:
        """
        From torch.nn.Linear (https://github.com/pytorch/pytorch/blob/af7dc23124a6e3e7b8af0637e3b027f3a8b3fb76/torch/nn/modules/linear.py#L103)
        StackOverflow discussion (https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch#:~:text=To%20initialize%20layers,do%20it%20afterwards)
        """
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)

        self.weight = torch.randn(self.out_features, self.in_features)
        self.bias   = torch.randn(self.out_features)

    def get_w_index_train(self, size):
        w_index = nn.Parameter(torch.randn(size**2, size**2))

        # w_index = self.get_w_index_space_translation(size)
        # w_index = nn.Parameter(w_index)
        # print('W_index trainable initialized as space_translation!')

        return w_index

    def get_w_index_space_translation(self, size):

        w1 = torch.zeros(size)
        w1[0] = 1.
        w2 = torch.stack([torch.roll(w1, shifts = shift) for shift in range(size)])
        w3 = torch.cat([torch.roll(w2, shifts = (shift, 0), dims = (0,1)) for shift in range(size)])

        w_index = torch.zeros(size**2, size**2)
        w_index[:, :size] = w3

        return w_index
                
    def get_w_index_none(self, size):
        out = torch.eye(size**2).reshape(size**2, size**2)
        return out
    
    def get_w_index_convolution(self, size):

        raise NotImplementedError

    def get_w_index_kernelconv(self, size):
        w_index = torch.zeros(size**2, size**2)

        w_index

        for s in range(size):
            w_index[s*size+s, s*size+s] = 1

        return w_index
    
    def normalize_w_index(self, w_index):
        # w_index = torch.abs(w_index)
        # w_index = torch.exp(w_index)
        w_index = w_index / torch.linalg.norm(w_index, ord = 1, dim = 1).reshape(-1, 1)

        # w_index_sum = torch.sum(w_index, dim = 1)
        # assert torch.allclose(w_index_sum, torch.ones_like(w_index_sum), atol=1e-5, rtol=1e-4), w_index_sum-torch.ones_like(w_index_sum)
        return w_index

    def forward(self, x, w_index = None):
        w_index = self.w_index if w_index is None else w_index
        
        # self.weight.data = self.weight.flatten()[w_index.flatten()].reshape(w_index.shape)
        # if self.set_bias:
        #     out =  x@self.weight.T + self.bias
        # else:
        #     out = x@self.weight.T
        # return out

        # indices = torch.max(self.w_index_param, dim = 0).indices
        # indices = torch.where(indices < self.out_features*self.in_features, indices, -1)

        # indices = self.w_index.flatten()

        # weight = self.weight.flatten()[indices].reshape(self.weight.shape)
        # weight = self.weight.flatten()[w_index.flatten()].reshape(w_index.shape)

        # weight = torch.where(indices.reshape(weight.shape) >= 0, weight, 0)

        if not all(torch.isin(torch.sum(w_index, dim = 1), torch.tensor([0., 1.]))):
            assert self.transform_type in ['train_Pw', 'trained']
            w_index = self.normalize_w_index(w_index)

        weight = (w_index @ self.weight.flatten()).reshape(self.weight.shape)



        # print(x.shape)
        # print(weight.shape)
        # print(x)
        # plt.imshow(weight.detach().numpy())
        # plt.show()

        # print(weight)


        if self.set_bias:
            out =  x@weight.T + self.bias
        else:
            out = x@weight.T
        return out
    
