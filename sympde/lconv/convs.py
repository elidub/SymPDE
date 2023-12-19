import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias = True, padding_mode='zeros'):
        super().__init__()
        assert padding_mode in ['zeros', 'circular']
        if padding_mode == 'zeros':
            padding_mode = 'constant'
        assert padding == (kernel_size - 1) // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.biases = nn.Parameter(torch.randn(out_channels))

    def einsum_alt(self, patches, weights):
        patches = patches.permute(0, 2, 1, 3) # # batch_size, width, channels, kernel_size 
        patches_unsqueezed = patches.unsqueeze(2) # batch_size, 1, width, channels, kernel_size
        weights_unsqueezed = weights.unsqueeze(0) # 1, out_channels, in_channels, kernel_size
        out = patches_unsqueezed * weights_unsqueezed
        out = out.sum([3, 4]) # batch_size, out_channels, width, channels
        out = out.permute(0, 2, 1) # batch_size, out_channels, output_pixels
        return out

    def forward(self, x):
        batch_size, in_channels2, width = x.shape
        assert in_channels2 == self.in_channels

        
        x_pad = F.pad(x, (self.padding, self.padding), mode=self.padding_mode)
        patches = x_pad.unsqueeze(2).unfold(3, self.kernel_size, 1)

        patches = patches.contiguous().view(batch_size, self.in_channels, width, self.kernel_size)

        self.patches = patches

        

        out = torch.einsum('biwk,oik->bow', patches, self.weights) # (biwk) -> (batch_size, in_channels, width, kernel)
        # out2 = self.einsum_alt(patches, self.weights)
        # out = out2
        # assert torch.allclose(out, out2)#, atol=1e-6, rtol=1e-6)

        # Add the bias
        if self.bias:
            out += self.biases.unsqueeze(0).unsqueeze(2)

        return out