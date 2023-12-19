import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, transform_type = 'none'):
        super().__init__()

        self.transform_type = transform_type

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features))


    def space_shift(self):
        assert self.out_features == self.in_features
        eye = torch.eye(self.out_features, self.in_features)
        w_index = torch.sum(torch.stack([torch.roll(eye*(i+1), shifts=shift, dims = 1) for i, shift in enumerate(range(self.in_features))]), dim=0)

        w = self.w.data
        b = self.b.data
        for p in torch.unique(w_index):
            idxs = torch.where(w_index == p)
            assert len(idxs) == 2
            idxs_val = idxs[0][0].item(), idxs[1][0].item() 
            w_new = w[idxs_val].clone() if p != 0 else 0.
            w[idxs] = w_new

            b_new = b[idxs_val[0]].clone() if p != 0 else 0.
            b[idxs[0]] = b_new

    def scaling(self):
        self.b.data = torch.zeros_like(self.b.data)

    def space_shift_scaling(self):
        # In this particular order
        self.space_shift()
        self.scaling()

    def transform(self, transform_type):
        if transform_type == 'none':
            pass
        elif transform_type == 'space_shift':
            self.space_shift()
        elif transform_type == 'scaling':
            self.scaling()
        elif transform_type == 'space_shift_scaling':
            self.space_shift_scaling()
        else:
            raise NotImplementedError

    def forward(self, x):
        # print(x.shape)
        self.transform(self.transform_type)
        if self.bias:
            out =  x@self.w.T + self.b
        else:
            out = x@self.w.T
        # print(out.shape)
        return out