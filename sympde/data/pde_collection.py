from scipy.fftpack import diff as psdiff
import numpy as np

class PDE_Pseudospectral:
    """
    Adapted from Brandstetter, J., Welling, M., Worrall, D.E., 2022. Lie Point Symmetry Data Augmentation for Neural PDE Solvers. https://doi.org/10.48550/arXiv.2202.07643
        https://github.com/brandstetter-johannes/LPSDA/blob/master/notebooks/data_augmentation.ipynb
    """
    def __init__(self, L):
        self.L = L

    def ux(self, u):
        return psdiff(u, order = 1, period=self.L)
    
    def uxx(self, u):
        return psdiff(u, order = 2, period=self.L)
    
    def uxxx(self, u):
        return psdiff(u, order = 3, period=self.L)
    
    def uxxxx(self, u):
        return psdiff(u, order = 4, period=self.L)
    
class CollectionPDE_Pseudospectral(PDE_Pseudospectral):
    def __init__(self, L):
        super().__init__(L)

        pde_collection = [self.pde1, self.pde2, self.pde3, self.pde4, self.pde5, self.pde6, self.pde7]
        self.collection = {f'pde{i+1}' : pde for i, pde in enumerate(pde_collection)}

        self.collection.update({'KdV' : self.KdV})


    def pde1(self, t, u):
        return 0.1 * self.uxx(u)
    
    def pde2(self, t, u):
        return 1. * self.uxx(u)
    
    def pde3(self, t, u):
        return 10. * self.uxx(u)
    
    def pde4(self, t, u):
        return self.ux(np.exp(u) * self.ux(u))
    
    def pde5(self, t, u):
        return np.exp(self.ux(u)) * self.uxx(u)
    
    def pde6(self, t, u):
        return (self.uxx(u) * np.exp(3 * np.arctan(self.ux(u)))) / (1 + self.ux(u)**2)
    
    def pde7(self, t, u):
        return np.arctan(self.uxx(u))
    
    def KdV(self, t, u):
        return - u * self.ux(u) - self.uxxx(u)

    
        
