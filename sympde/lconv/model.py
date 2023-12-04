import torch
import torch.nn as nn

from lconv.mnist import Reshape

class Lconv_core(nn.Module):
    """ L-conv layer with full L """
    def __init__(self,d,num_L=1,cin=1,cout=1):
        """
        L:(num_L, d, d)
        Wi: (num_L, cout, cin)
        """
        super().__init__()
        self.L = nn.Parameter(torch.Tensor(num_L, d, d))
        self.Wi = nn.Parameter(torch.Tensor(num_L, cout, cin))
        
        # initialize weights and biases
        nn.init.kaiming_normal_(self.L) 
        nn.init.kaiming_normal_(self.Wi)
                
    def forward(self, x):
        # x:(batch, channel, flat_d)
        # h = (x + Li x Wi) W0
        y = torch.einsum('kdf,bcf,koc->bod', self.L, x, self.Wi ) +x #+ self.b        
        return y

class LconvNet(nn.Module):
    def __init__(self, shape):
        super().__init__()

        d = torch.prod(shape)

        lc = Lconv_core(d=d, num_L=1, cin=1, cout=1)

        self.layers = nn.Sequential(
            nn.Flatten(2),
            lc,
            Reshape(shape),
        )

    def forward(self, x):
        return self.layers(x)
