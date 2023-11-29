import torch
import inspect 

from data.pseudospectral import PDE_Pseudospectral
from data.augment import fourier_shift

def PDEs():
    return {
        "Pde1": Pde1(),
        "KdV": KdV(),
    }

class BasePDE(PDE_Pseudospectral):
    def __init__(self):
        super().__init__()
        self.n_augments = len([func_name for func_name, _ in inspect.getmembers(self, predicate=inspect.ismethod) if func_name.startswith('_u')]) + 1

    def __repr__(self) -> str:
        return self.__class__.__name__

    def augment(self, u, x, t, epss):
        raise NotImplementedError
    
    def __call__(self, t, u, L):
        raise NotImplementedError



class Pde1(BasePDE):
    def __init__(self):
        super().__init__()

    def __call__(self, t, u, L):
        return 0.1 * self.uxx(u, L)
    
class KdV(BasePDE):
    def __init__(self):
        super().__init__()
        self.n_augments = 4

    def _u2(self, u, x, t, eps):
        """
        Space Translate
        """
        # print(eps)
        x_new = x
        t_new = t
        # u_new = fourier_shift(u, eps = -eps, dim = -1)
        u_new = (fourier_shift(u.unsqueeze(0), eps, dim = -1)).squeeze(0)
        return u_new, x_new, t_new

    def _u3(self, u, x, t, eps):
        """
        Galileo
        """

        dx = x[0,1] - x[0, 0]
        Nt, Nx = u.shape
        L = dx * Nx

        d = - eps * (t[:, 0] / L) # minus, t, L are all necessary
        eps_u = - eps
        x_new = x
        t_new = t
        u_new = (fourier_shift(u, eps=d[:, None], dim=-1) + eps_u).squeeze(0) # minus is necassary
        return u_new, x_new, t_new

    def _u4(self, u, x, t, eps):
        """
        Scaling
        """
        eps_u = - eps
        x_new = x * torch.exp(-eps)
        t_new = t * torch.exp(-3 * eps)
        u_new = u * torch.exp(- 2 * eps_u)
        return u_new, x_new, t_new

    def augment(self, u, x, t, epsilons):
        u_new, x_new, t_new = u, x, t
        u_new, x_new, t_new = self._u2(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * epsilons[1]) #  space translate
        u_new, x_new, t_new = self._u3(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * epsilons[2]) # scaling
        u_new, x_new, t_new = self._u4(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 2. * epsilons[3]) # galileo
        return u_new, x_new, t_new

    def __call__(self, t, u, L):
        return - u * self.ux(u, L) - self.uxxx(u, L)