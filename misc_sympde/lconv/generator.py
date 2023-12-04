import numpy as np
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Initialize 6 trainable parameters for basis coefficients:
        self.a = nn.Parameter(torch.randn(6), requires_grad=True)
        self.D = nn.Parameter(self.generate_basis(), requires_grad=False)
        self.D_titles = ['Lx', 'xLx', 'yLx', 'Ly', 'xLy', 'yLy']
        # self.G = torch.einsum('i, imn -> mn', self.a, self.D)

        self.Lx, self.xLx, self.yLx, self.Ly, self.xLy, self.yLy = self.D


    def generate_basis(self):
        L0 = lambda d,z: np.sum([2*np.pi*p/d**2 * np.sin(2*np.pi*p/d *z) 
                                    for p in np.arange(-d/2+1,d/2)], axis=0)
        d = self.dim
        coords = np.mgrid[:d,:d] - d/2
        x,y = coords.reshape((2,-1))

        dx = (x[:,np.newaxis] - x) * (y[:,np.newaxis] == y)
        dy = (y[:,np.newaxis] - y) * (x[:,np.newaxis] == x)

        Lx = L0(2*d, dx)
        Ly = L0(2*d, dy)

        xLx = np.diag(x) @ Lx
        yLx = np.diag(y) @ Lx
        xLy = np.diag(x) @ Ly
        yLy = np.diag(y) @ Ly

        D = np.stack([Lx, xLx, yLx, Ly, xLy, yLy], axis=0)
        D = torch.from_numpy(D).float()
        return D


# gen = Generator(25)
# D = c.generate_basis()

# len_D = len(D)

# fig, axs = plt.subplots(2, len_D//2, figsize=(6, 4), tight_layout=True)
# axs = axs.flatten()
# for i, d in enumerate(D):
#     axs[i].imshow(d)
#     axs[i].set_title(f'D_{i}')
#     axs[i].axis('off')
# plt.show()

# c.a.shape, c.D.shape