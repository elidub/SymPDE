import torch
import numpy as np
import inspect 

from data.pseudospectral import PDE_Pseudospectral
from data.augment import fourier_shift

def PDEs():
    return {
        "Pde1": Pde1(),
        "Pde2": Pde2(),
        "Pde3": Pde3(),
        "Pde4": Pde4(),
        "Pde5": Pde5(),
        "Pde6": Pde6(),
        "Pde7": Pde7(),
        "Pde8": Pde8(),
        "Pde9": Pde9(),
        "Pde10": Pde10(),
        "Pde11": Pde11(),
        "Pde12": Pde12(),
        "Pde13": Pde13(),
        "Pde14": Pde14(),
        "Pde15": Pde15(),
        "Pde16": Pde16(),
        "Pde17": Pde17(),
        "Pde18": Pde18(),
        "Pde19": Pde19(),
        "Pde20": Pde20(),
        "Pde21": Pde21(),
        "Pde22": Pde22(),
        "Pde23": Pde23(),
        "Pde24": Pde24(),
        "KdV": KdV(),
    }

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


class Pde1(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 100

    def __str__(self) -> str:
        return r"$0.1 u_{xx}$"
    
    def _u1(self, u, x, t, eps):
        return u, x, t

    def _u2(self, u, x, t, eps):
        """
        Space Translate
        """
        x_new = x
        t_new = t
        # u_new = fourier_shift(u, eps = -eps, dim = -1)
        u_new = (fourier_shift(u.unsqueeze(0), eps, dim = -1)).squeeze(0)
        return u_new, x_new, t_new

    # def _u3(self, u, x, t, eps):
    #     """
    #     Galileo
    #     """

    #     dx = x[0,1] - x[0, 0]
    #     Nt, Nx = u.shape
    #     L = dx * Nx

    #     d = - eps * (t[:, 0] / L) # minus, t, L are all necessary
    #     eps_u = - eps
    #     x_new = x
    #     t_new = t
    #     u_new = (fourier_shift(u, eps=d[:, None], dim=-1) + eps_u).squeeze(0) # minus is necassary
    #     return u_new, x_new, t_new

    def _u3(self, u, x, t, eps):
        """
        Scaling
        """
        eps_u = - eps
        x_new = x * torch.exp(-eps)
        t_new = t * torch.exp(-2 * eps)
        u_new = u 
        return u_new, x_new, t_new
    
    def _u6(self, u, x, t, eps):
        """
        Scaling (6)
        """
        eps_u = - eps
        x_new = x
        t_new = t
        u_new = u * torch.exp(eps_u)
        return u_new, x_new, t_new

    def __call__(self, t, u, L):
        return 0.1 * self.dxx(u, L)
    
class Pde2(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 100

    def __str__(self) -> str:
        return r"$1.0 u_{xx}$"

    def __call__(self, t, u, L):
        return 1.0 * self.dxx(u, L)
    
class Pde3(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 100

    def __str__(self) -> str:
        return r"$10.0 u_{xx}$"

    def __call__(self, t, u, L):
        return 10 * self.dxx(u, L)
    
class Pde4(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 100

    def __str__(self) -> str:
        return r"$(e^{u_x} u_{x})_{x}$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L)
    
class Pde5(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 100

    def __str__(self) -> str:
        return r"$e^{u_x} u_{x x}$"

    def __call__(self, t, u, L):
        return np.exp(self.dx(u, L)) * self.dxx(u, L)
    
class Pde6(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 200

    def __str__(self) -> str:
        return r"$\frac{u_{x x}}{\left(u_x\right)^2+1} e^{3 \arctan \left(u_x\right)}$"

    def __call__(self, t, u, L):
        return (self.dxx(u, L) * np.exp(3 * np.arctan(self.dx(u, L)))) / (1 + self.dx(u, L)**2)

class Pde7(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 200

    def __str__(self) -> str:
        return r"$\arctan \left(u_{x x}\right)$"

    def __call__(self, t, u, L):
        return np.arctan(self.dxx(u, L))
    
class Pde8(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$\left(e^u u_x\right)_x+e^{-2 u}$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) + np.exp(-2*u)
    
class Pde9(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2
        
    def __str__(self) -> str:
        return r"$\left(e^u u_x\right)_x+e^{-u}$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) + np.exp(-u)
    
class Pde10(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$\left(e^u u_x\right)_x-e^u$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) - np.exp(u)
    
class Pde11(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$\left(e^u u_x\right)_x-e^{2 u}$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) - np.exp(2*u)
    
class Pde12(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$\left(e^u u_x\right)_x+1$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) + 1
    
class Pde13(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$\left(e^u u_x\right)_x-1$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) - 1
    
class Pde14(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$u_{x x}-e^u$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) - np.exp(u)
    
class Pde15(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 200

    def __str__(self) -> str:
        return r"$u_{x x}+u^{-1}$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) + u**(-1)
    
class Pde16(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 0.1

    def __str__(self) -> str:
        return r"$u_{x x}+u^2$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) + u**(2)
    
class Pde17(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 0.1
        
    def __str__(self) -> str:
        return r"$u_{x x}-u^2$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) - u**(2)
    
class Pde18(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$u_{x x}+u$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) + u

class Pde19(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$u_{x x}-u$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) - u

class Pde20(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2

    def __str__(self) -> str:
        return r"$u_{x x}+1$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) + 1
    
class Pde21(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 2
        
    def __str__(self) -> str:
        return r"$u_{x x}-1$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) - 1
    
class Pde22(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 200

    def __str__(self) -> str:
        return r"$u u_x+u_{x x}$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) + u * self.dx(u, L)
    
class Pde23(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 200

    def __str__(self) -> str:
        return r"$u_{x x}+\left(u_x\right)^2$"

    def __call__(self, t, u, L):
        return self.dxx(u, L) + self.dx(u, L)**2
    
class Pde24(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 200
        
    def __str__(self) -> str:
        # return r"$u u_x+u_{x x}$"
        return r"$\left(e^u u_x\right)_x-u u_x$"

    def __call__(self, t, u, L):
        return self.dx(np.exp(u) * self.dx(u, L), L) - u * self.dx(u, L)

    
class KdV(BasePDE):
    def __init__(self):
        super().__init__()
        self.Tmax = 50

    def __str__(self) -> str:  
        return r"$- u u_x - u_{xxx}$"
    
    def _u1(self, u, x, t, eps):
        return u, x, t

    def _u2(self, u, x, t, eps):
        """
        Space Translate
        """
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
        eps = 2 * eps
        eps_u = - eps
        x_new = x * torch.exp(-eps)
        t_new = t * torch.exp(-3 * eps)
        u_new = u * torch.exp(- 2 * eps_u)
        return u_new, x_new, t_new

    # def augment(self, u, x, t, epsilons):
    #     u_new, x_new, t_new = u, x, t
    #     # u_new, x_new, t_new = self._u2(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * epsilons[1]) #  space translate
    #     # u_new, x_new, t_new = self._u3(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * epsilons[2]) # scaling
    #     # u_new, x_new, t_new = self._u4(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 2. * epsilons[3]) # galileo
    #     u_new, x_new, t_new = self._u2(u_new, x_new, t_new, eps = torch.tensor(epsilons[1])) #  space translate
    #     u_new, x_new, t_new = self._u3(u_new, x_new, t_new, eps = torch.tensor(epsilons[2])) # galileo 
    #     u_new, x_new, t_new = self._u4(u_new, x_new, t_new, eps = torch.tensor(2. * epsilons[3])) # scaling
 
    #     return u_new, x_new, t_new

    def __call__(self, t, u, L):
        return - u * self.dx(u, L) - self.dxxx(u, L)