import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Conv1dMag(nn.Module):
    def __init__(self, 
                 input_channels,
                 output_channels, 
                 kernel_size, 
                 activation = True, # whether to use activation functions
                 stride = 1, 
                 deconv = False, # Whether this is used as a deconvolutional layer
                 padding_mode='circular',
        ):
        """
        Magnitude Equivariant 2D Convolutional Layers
        """
        # super(mag_conv2d, self).__init__()
        super().__init__()
        self.activation = activation
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride = kernel_size, padding_mode=self.padding_mode, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.deconv = deconv
        self.output_channels = output_channels
        self.eps = 10e-10
        
    def unfold(self, x):
        """
        Extracts sliding local blocks from a batched input tensor.
        """
        batch_size, time, space = x.shape
        assert self.input_channels == time

        if not self.deconv:
            x = F.pad(x, ((self.pad_size,)*2), mode = self.padding_mode)
        
        x = x.reshape(batch_size, self.input_channels, space+2*self.pad_size, 1)
        
        x = F.unfold(x, kernel_size = (self.kernel_size, 1))
        assert x.shape[-1] == space
        
        x = x.reshape(batch_size, self.input_channels, self.kernel_size, space)
        
        if self.stride > 1:
            x = x[:,:,:,::self.stride]
            
        return x
    
    def transform(self, x):
        """
        Max-Min Normalization on each sliding local block.
        """   
        batch_size, time, kernel, space = x.shape
        
        channel_idx, kernel_idx = 1, 2
        scaler = (x.max(channel_idx).values.unsqueeze(channel_idx).max(kernel_idx).values.unsqueeze(kernel_idx) - 
                x.min(channel_idx).values.unsqueeze(channel_idx).min(kernel_idx).values.unsqueeze(kernel_idx))
        
        x = x /(scaler + self.eps) 
        
        scaler = scaler.squeeze(2)

        x = x.reshape(batch_size, self.input_channels, self.kernel_size, space)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch_size, self.input_channels, space*self.kernel_size)
        return x, scaler
    
    
    def inverse_transform(self, x, scaler):
        """
        Inverse Max-Min Normalization.
        """   
        x = x * (scaler + self.eps)
        return x
    
    def forward(self, x):
        
        x = self.unfold(x)
        x, stds = self.transform(x)
        x = self.conv1d(x)
        
        if self.activation:
            x = F.relu(x)
        
        x = self.inverse_transform(x, stds)
        return x



class Conv2dMag(nn.Module):
    def __init__(self, 
                 input_channels,
                 output_channels, 
                 kernel_size, 
                 activation = True, # whether to use activation functions
                 stride = 1, 
                 deconv = False):# Whether this is used as a deconvolutional layer
        """
        Magnitude Equivariant 2D Convolutional Layers
        """
        # super(mag_conv2d, self).__init__()
        super().__init__()
        self.activation = activation
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.deconv = deconv
        self.output_channels = output_channels
        self.eps = 10e-7
        
    def unfold(self, x):
        """
        Extracts sliding local blocks from a batched input tensor.
        """
        batch_size, input_channels, height, width = x.shape
        assert self.input_channels == input_channels
        # print('in', x.shape)
        if not self.deconv:
            x = F.pad(x, ((self.pad_size, self.pad_size)*2), mode = 'replicate')
        # print('padded', x.shape)
        out = F.unfold(x, kernel_size = (self.kernel_size,self.kernel_size))
        # print('unfold', out.shape)
        assert (int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1]))) == (height, width)
        out = out.reshape(batch_size, self.input_channels, self.kernel_size, self.kernel_size, height, width)
        # print(out.shape)
        
        if self.stride > 1:
            out = out[:,:,:,:,::self.stride,::self.stride]
            
        return out
    
    def transform(self, x):
        """
        Max-Min Normalization on each sliding local block.
        """   
        batch_size, input_channels, kernel_x, kernel_y, height, width = x.shape
        
        out = x
        # print('in', out.shape)
        channel_idx, kernel_x_idx, kernel_y_idx = 1, 2, 3
        stds = (out.max(channel_idx).values.unsqueeze(channel_idx).max(kernel_x_idx).values.unsqueeze(kernel_x_idx).max(kernel_y_idx).values.unsqueeze(kernel_y_idx) - 
                out.min(channel_idx).values.unsqueeze(channel_idx).min(kernel_x_idx).values.unsqueeze(kernel_x_idx).min(kernel_y_idx).values.unsqueeze(kernel_y_idx))
        
        out = out /(stds + self.eps) 
        
        # print('stds', stds.shape)
        stds = stds.squeeze(3).squeeze(3).squeeze(2)
        # print('stds', stds.shape)

        # print('divided', out.shape)
        out = out.reshape(batch_size, self.input_channels, self.kernel_size, self.kernel_size, height, width)
        # print('reshaped', out.shape)
        # print('transpose', out.shape)
        out = out.permute(0, 1, 4, 3, 5, 2)
        # print('transposed', out.shape)
        out = out.reshape(batch_size, self.input_channels, height*self.kernel_size, width*self.kernel_size)
        # print('reshaped', out.shape)
        return out, stds
    
    
    def inverse_transform(self, out, stds):
        """
        Inverse Max-Min Normalization.
        """   
        # batch_size, input_channels, height, width = out.shape
        print('in', out.shape)
        print('stds', stds.shape)
        # out = out.reshape(batch_size, input_channels, height, width)
        # print('reshaped', out.shape)
        out = out * (stds + self.eps)
        # print('scaled', out.shape)
        # out = out.reshape(batch_size, -1, height, width)
        # print('reshaped', out.shape)
        return out
    
    def forward(self, x):
        x_org = x.clone()
        # print(x.shape)
        x = self.unfold(x)
        # print(x.shape)
        x, stds = self.transform(x)
        # print(x.shape, stds.shape)
        # out = self.conv2d(x)
        i, j = stds.shape[-2:]
        out = x[:, :self.output_channels, :i, :j]
        # print(out.shape)
        # assert (out.shape == x_org.shape), (out.shape, x_org.shape)
        if self.activation:
            out = F.relu(out)
            # print('activation', out.shape)
        out = self.inverse_transform(out, stds)
        # print(out.shape)
        return out
