import numpy as np
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim/2 == dim//2

        # Initialize 6 trainable parameters for basis coefficients:
        self.a = nn.Parameter(torch.randn(6), requires_grad=True)
        self.D = nn.Parameter(self.generate_basis(), requires_grad=False)
        self.D_titles = [r'$L_x$', r'$xL_x$', r'$yL_x$', r'$L_y$', r'$xL_y$', r'$yL_y$']
        # self.G = torch.einsum('i, imn -> mn', self.a, self.D)

        self.Lx, self.xLx, self.yLx, self.Ly, self.xLy, self.yLy = self.D


    def generate_basis(self):
        L0 = lambda d,z: np.sum([2*np.pi*p/d**2 * np.sin(2*np.pi*p/d *z) 
                                    for p in np.arange(-d/2+1,d/2)], axis=0)
        d = self.dim
        coords = np.mgrid[:d,:d] - d/2
        y, x = coords.reshape((2,-1)) # x, y

        self.x, self.y = x, y

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
    
class UGenerator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim/2 == dim//2

        self.generate_basis()

        # Initialize 6 trainable parameters for basis coefficients:
        # self.a = nn.Parameter(torch.randn(6), requires_grad=True)
        # self.D = nn.Parameter(self.generate_basis(), requires_grad=False)
        # self.D_titles = [r'$L_x$', r'$xL_x$', r'$yL_x$', r'$L_y$', r'$xL_y$', r'$yL_y$']
        # # self.G = torch.einsum('i, imn -> mn', self.a, self.D)

        # self.Lx, self.xLx, self.yLx, self.Ly, self.xLy, self.yLy = self.D


    def generate_basis(self):
        L0 = lambda d,z: np.sum([2*np.pi*p/d**2 * np.sin(2*np.pi*p/d *z) 
                                    for p in np.arange(-d/2+1,d/2)], axis=0)
        d = self.dim
        coords = np.mgrid[:d,:d] - d/2
        y, x = coords.reshape((2,-1)) # x, y

        self.x, self.y = np.diag(x), np.diag(y)

        dx = (x[:,np.newaxis] - x) * (y[:,np.newaxis] == y)
        dy = (y[:,np.newaxis] - y) * (x[:,np.newaxis] == x)

        self.Lx = L0(2*d, dx)
        self.Ly = L0(2*d, dy)
        self.Lu = np.diag(np.full((d**2), -1)) # Let's put -1 as a 'flag' for Lu. Later we can use this to check if the matrix has Lu or not.

        self.xLx = np.diag(x) @ self.Lx
        self.yLx = np.diag(y) @ self.Lx
        self.xLy = np.diag(x) @ self.Ly
        self.yLy = np.diag(y) @ self.Ly
        self.uLu = np.eye(d**2)



class HardCodedGenerator:
    def __init__(self, dim: int):

        Ly = self.get_Ly(dim)
        Lx = self.get_Lx(dim) 
        self.x, self.y = self.get_xy(dim)

        xLx = np.diag(self.x) @ Lx
        yLx = np.diag(self.y) @ Lx
        xLy = np.diag(self.x) @ Ly
        yLy = np.diag(self.y) @ Ly

        self.D = np.stack([Lx, xLx, yLx, Ly, xLy, yLy], axis=0)
        self.Lx, self.xLx, self.yLx, self.Ly, self.xLy, self.yLy = self.D

    def cycle(self,x, i):
        return np.concatenate([x[i:], x[:i]])
    
    def get_Ly(self, dim):
        diag = np.eye(dim**2)
        return self.cycle(diag, dim) + -self.cycle(diag, -dim)
    
    def get_Lx(self, dim):
        x0 = np.zeros(dim)
        x0[1], x0[-1] = 1, -1
        x = np.zeros((dim, dim))
        for i in range(dim):
            x[i] = np.roll(x0, i)

        y = np.zeros((dim**2, dim**2))

        for d in range(dim):
            s = slice(d*dim, (d+1)*dim)
            y[s, s] = x

        return y
    
    def get_xy(self, dim):
        d = dim
        coords = np.mgrid[:d,:d] - d/2
        y, x = coords.reshape((2,-1)) # x, y
        return x, y