"""
Magnitude Equivariant ResNet and U-net
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mag_conv2d(nn.Module):
    def __init__(self, 
                 input_channels,
                 output_channels, 
                 kernel_size, 
                 um_dim = 2, # The number of channels of each frame
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
        self.um_dim = um_dim 
        self.stride = stride
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.deconv = deconv
        self.output_channels = output_channels

    def unfold(self, x):
        """
        Extracts sliding local blocks from a batched input tensor.
        """
        # print('in', x.shape)
        if not self.deconv:
            x = F.pad(x, ((self.pad_size, self.pad_size)*2), mode = 'replicate')
            # print('padded')
        out = F.unfold(x, kernel_size = self.kernel_size)
        out2 = F.unfold(x, kernel_size = (3,3))
        assert (out == out2).all()
        # print('unfolded', out.shape, 'with kernel size', self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, out.shape[-1])
        out2 = out2.reshape(out2.shape[0], self.input_channels, self.kernel_size, self.kernel_size, int(np.sqrt(out2.shape[-1])), int(np.sqrt(out2.shape[-1])))      
        # print('reshaped', out.shape)
        ## Batch_size x (in_channels x kernel_size x kernel_size) x 64 x 64
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1])))      
        assert (out == out2).all()
        # print('reshaped again', out.shape)
        if self.stride > 1:
            out = out[:,:,:,:,::self.stride,::self.stride]
            # print('strided', out.shape)
            return out
            
        return out
    
    def transform(self, x):
        """
        Max-Min Normalization on each sliding local block.
        """   
        # print('in', x.shape)
        # Calculates the max-min of input sliding local blocks 
        out = x.reshape(x.shape[0], self.input_channels//self.um_dim, self.um_dim, self.kernel_size, self.kernel_size, x.shape[-2], x.shape[-1])
        # print('reshaped', out.shape)
        stds = (out.max(1).values.unsqueeze(1).max(3).values.unsqueeze(3).max(4).values.unsqueeze(4) - 
                out.min(1).values.unsqueeze(1).min(3).values.unsqueeze(3).min(4).values.unsqueeze(4))
        
        # print('stds', stds.shape)

        out = out /(stds + 10e-7) 
        # print('divided', out.shape)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, x.shape[-2], x.shape[-1])
        # print('reshaped', out.shape)
        out = out.transpose(2,4).transpose(-1,-2)
        # print('transpose', out.shape)
        out2 = out.reshape(out.shape[0], self.input_channels, x.shape[-2]*self.kernel_size, x.shape[-1]*self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, x.shape[-2]*self.kernel_size, x.shape[-1], self.kernel_size)
        # print('reshaped again', out.shape)

        ## Batch_size x in_channels x (64 x kernel_size) x (64 x kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, x.shape[-2]*self.kernel_size, x.shape[-1]*self.kernel_size)
        # print('reshaped again again', out.shape)
        assert (out == out2).all()
        return out, stds.squeeze(3).squeeze(3)
    
    
    def inverse_transform(self, out, stds):
        """
        Inverse Max-Min Normalization.
        """   
        # print('in', out.shape)
        out = out.reshape(out.shape[0], out.shape[1]//self.um_dim, self.um_dim, out.shape[-2], out.shape[-1])
        # print('reshaped', out.shape)
        out = out * (stds + 10e-7)
        # print('scaled', out.shape)
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        # print('reshaped', out.shape)
        return out
    
    def forward(self, x):
        # x_org = x.clone()
        # print(x.shape)
        x = self.unfold(x)
        # print(x.shape)
        x, stds = self.transform(x)
        # print(x.shape, stds.shape)
        out = self.conv2d(x)
        # i, j = stds.shape[-2:]
        # out = x[:, :self.output_channels, :i, :j]
        # print(out.shape)
        # assert (out.shape == x_org.shape), (out.shape, x_org.shape)
        if self.activation:
            out = F.relu(out)
            # print('activation', out.shape)
        out = self.inverse_transform(out, stds)
        # print(out.shape)
        return out

    
# Magnitude Equivariant ResNet.   
class mag_resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(mag_resblock, self).__init__()
        self.layer1 = mag_conv2d(input_channels, hidden_dim, kernel_size)
        self.layer2 = mag_conv2d(hidden_dim, hidden_dim, kernel_size)
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        if input_channels != hidden_dim:
            self.upscale = mag_conv2d(input_channels, hidden_dim, kernel_size, activation = False)
        
    def forward(self, x):
        out = self.layer1(x)
        
        if self.input_channels != self.hidden_dim:
            out = self.layer2(out) + self.upscale(x)
        else:
            out = self.layer2(out) + x
        print('forward', out.shape)
        return out

class ResNet_Mag(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        # super(ResNet_Mag, self).__init__()
        super().__init__()
        layers = [mag_resblock(input_channels, 64, kernel_size), mag_resblock(64, 64, kernel_size)]
        layers += [mag_resblock(64, 128, kernel_size), mag_resblock(128, 128, kernel_size)]
        layers += [mag_resblock(128, 256, kernel_size), mag_resblock(256, 256, kernel_size)]
        layers += [mag_resblock(256, 512, kernel_size), mag_resblock(512, 512, kernel_size)]
        layers += [mag_conv2d(512, output_channels, kernel_size = kernel_size, activation = False)]
        self.model = nn.Sequential(*layers)
             
    def forward(self, x):
        print('x', x.shape)
        out = self.model(x)
        print('out', out.shape)
        return out

    
class mag_deconv2d(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Magnitude Equivariant 2D Transposed Convolutional Layers
        """
        super(mag_deconv2d, self).__init__()
        self.conv2d = mag_conv2d(input_channels = input_channels, output_channels = output_channels, kernel_size = 4, um_dim = 2,
                             activation = True, stride = 1, deconv = True)
    
    def pad(self, x):
        pad_x = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2)
        pad_x[:,:,::2,::2].copy_(x)
        pad_x = F.pad(pad_x, (1,2,1,2), mode='replicate')
        return pad_x
    
    def forward(self, x):
        out = self.pad(x).to(device)
        return self.conv2d(out)

# Magnitude Equivariant U_net.   
class Unet_Mag(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Unet_Mag, self).__init__()
        self.input_channels = input_channels
        self.conv1 = mag_conv2d(input_channels, 64, kernel_size = kernel_size, stride=2)
        self.conv1_1 = mag_conv2d(64, 64, kernel_size = kernel_size, stride=1)
        self.conv2 = mag_conv2d(64, 128, kernel_size = kernel_size, stride=2)
        self.conv2_1 = mag_conv2d(128, 128, kernel_size = kernel_size, stride = 1)
        self.conv3 = mag_conv2d(128, 256, kernel_size = kernel_size, stride=2)
        self.conv3_1 = mag_conv2d(256, 256, kernel_size = kernel_size, stride=1)
        self.conv4 = mag_conv2d(256, 512, kernel_size = kernel_size, stride=2)
        self.conv4_1 = mag_conv2d(512, 512, kernel_size = kernel_size, stride=1)

        self.deconv3 = mag_deconv2d(512, 128)
        self.deconv2 = mag_deconv2d(384, 64)
        self.deconv1 = mag_deconv2d(192, 32)
        self.deconv0 = mag_deconv2d(96, 16)
        self.output_layer = mag_conv2d(16 + input_channels, output_channels, kernel_size=kernel_size, activation = False)

    def forward(self, x):
        out_conv1 = self.conv1_1(self.conv1(x))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        out_deconv3 = self.deconv3(out_conv4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)
        return out




    
