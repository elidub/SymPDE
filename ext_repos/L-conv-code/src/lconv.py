import torch
import torch.nn as nn


class Lconv(nn.Module):
    """ L-conv layer with full L """
    def __init__(self,d,num_L=1,cin=1,cout=1,rank=8):
        """
        L:(num_L, d, d)
        Wi: (num_L, cout, cin)
        """
        super().__init__()
        self.L = nn.Parameter(torch.Tensor(num_L, d, d))
        self.Wi = nn.Parameter(torch.Tensor(num_L+1, cout, cin)) #  W^0 = Wi[0], W^0\epsion^i = Wi[1:] 
        
        # initialize weights and biases
        nn.init.kaiming_normal_(self.L) 
        nn.init.kaiming_normal_(self.Wi)
                
    def forward(self, x):
        # x:(batch, channel, flat_d)
        # res = x W0
        residual = torch.einsum('bcd,oc->bod', x, self.Wi[0] )
        print('x',x.shape, 'Wi[0]', self.Wi[0].shape, 'residual', residual.shape)
        # y = Li x Wi
        y = torch.einsum('kdf,bcf,koc->bod', self.L, x, self.Wi[1:])
        print('L',self.L.shape, 'x', x.shape, 'Wi[1:]', self.Wi[1:].shape, 'y', y.shape)
        return y + residual
    
class Reshape(nn.Module):
    def __init__(self,shape=None):
        self.shape = shape
        super().__init__()
    def forward(self,x):
        return x.view(-1,*self.shape)
    
