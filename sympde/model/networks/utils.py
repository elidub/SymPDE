import logging
import torch
import torch.nn as nn

class PrintLayer(nn.Module):
    def __init__(self, print_str = 'shape'):
        super().__init__()
        self.print_str = print_str
    
    def forward(self, x):
        # logging.debug(f'{self.print_str}: {x.shape}')
        print(f'{self.print_str}: {x.shape}')
        return x