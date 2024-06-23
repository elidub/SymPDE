from scipy.fftpack import diff as psdiff
import numpy as np
import inspect 

class PDE_Pseudospectral:
    """
    Adapted from Brandstetter, J., Welling, M., Worrall, D.E., 2022. Lie Point Symmetry Data Augmentation for Neural PDE Solvers. https://doi.org/10.48550/arXiv.2202.07643
        https://github.com/brandstetter-johannes/LPSDA/blob/master/notebooks/data_augmentation.ipynb
    """
    def __init__(self):
        pass

    def dx(self, u, L):
        return psdiff(u, order = 1, period=L)
    
    def dxx(self, u, L):
        return psdiff(u, order = 2, period=L)
    
    def dxxx(self, u, L):
        return psdiff(u, order = 3, period=L)
    
    def dxxxx(self, u, L):
        return psdiff(u, order = 4, period=L)
    
class BasePDE(PDE_Pseudospectral):
    def __init__(self):
        super().__init__()
        self.aug_methods = [method for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod) if method_name.startswith('_u')]

        self.Lmax = None
        self.Tmax = None

    def __repr__(self) -> str:
        return self.__class__.__name__
    
    def __str__(self) -> str:
        raise NotImplementedError

    def __call__(self, t, u, L):
        raise NotImplementedError

    