from scipy.fftpack import diff as psdiff
import numpy as np

class PDE_Pseudospectral:
    """
    Adapted from Brandstetter, J., Welling, M., Worrall, D.E., 2022. Lie Point Symmetry Data Augmentation for Neural PDE Solvers. https://doi.org/10.48550/arXiv.2202.07643
        https://github.com/brandstetter-johannes/LPSDA/blob/master/notebooks/data_augmentation.ipynb
    """
    def __init__(self):
        pass

    def ux(self, u, L):
        return psdiff(u, order = 1, period=L)
    
    def uxx(self, u, L):
        return psdiff(u, order = 2, period=L)
    
    def uxxx(self, u, L):
        return psdiff(u, order = 3, period=L)
    
    def uxxxx(self, u, L):
        return psdiff(u, order = 4, period=L)
    