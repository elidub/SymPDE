import torch
import numpy as np

from data.lpda_data_aug import fourier_shift

def pde1_u2(u, x, t, eps):
    # print(eps)
    x_new = x
    t_new = t
    # u_new = fourier_shift(u, eps = -eps, dim = -1)
    u_new = (fourier_shift(u.unsqueeze(0), eps, dim = -1)).squeeze(0)
    return u_new, x_new, t_new

def pde1_u3(u, x, t, eps):
    # print(eps)
    x_new = x * torch.exp(-eps)
    t_new = t * torch.exp(-2 * eps)
    u_new = u
    return u_new, x_new, t_new

def pde1_u4(u, x, t, eps):
    # print(eps)
    dx = x[0,1] - x[0, 0]
    Nt, Nx = u.shape
    L = dx * Nx


    d = - 2*eps * (t[:, 0] / L)
    eps_u = - eps
    x_new = x
    t_new = t

    u_mult = torch.exp(-eps_u * (x + -eps_u * t) / 0.1)
    u_new = (fourier_shift(u, eps=d[:, None], dim=-1)).squeeze(0) * u_mult
    return u_new, x_new, t_new

def pde1_u5(u, x, t, eps):
    # print(eps)

    eps = np.abs(eps)

    eps_u = - eps
    shift1 = 1 + 4*eps * t
    shift1_u = 1 + 4*-eps_u * t
    u_mult = torch.exp( (-eps * x**2) / (0.1 * shift1_u) ) / torch.sqrt(shift1_u)

    x_new = x / shift1 
    t_new = t / shift1 

    u_new = u * u_mult
    return u_new, x_new, t_new

def pde1_u6(u, x, t, eps):
    # print(eps)
    eps_u = - eps
    x_new = x
    t_new = t
    u_new = torch.exp(eps_u) * u
    return u_new, x_new, t_new

def augment_pde1(u_new, x_new, t_new):

    u_new, x_new, t_new = pde1_u2(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 3 ) #  
    u_new, x_new, t_new = pde1_u3(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 0.1) # have to be low
    u_new, x_new, t_new = pde1_u4(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 0.01) # have to be low
    u_new, x_new, t_new = pde1_u5(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 0.001) # kinda okish
    u_new, x_new, t_new = pde1_u6(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 1.) # 

    return u_new, x_new, t_new




def KdV_u2(u, x, t, eps):
    """
    Space Translate
    """
    # print(eps)
    x_new = x
    t_new = t
    # u_new = fourier_shift(u, eps = -eps, dim = -1)
    u_new = (fourier_shift(u.unsqueeze(0), eps, dim = -1)).squeeze(0)
    return u_new, x_new, t_new

def KdV_u3(u, x, t, eps):
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

def KdV_u4(u, x, t, eps):
    """
    Scaling
    """
    eps_u = - eps
    x_new = x * torch.exp(-eps)
    t_new = t * torch.exp(-3 * eps)
    u_new = u * torch.exp(- 2 * eps_u)
    return u_new, x_new, t_new

def augment_KdV(u_new, x_new, t_new):

    u_new, x_new, t_new = KdV_u2(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 1.) #  space translate
    u_new, x_new, t_new = KdV_u4(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 0.1) # scaling
    u_new, x_new, t_new = KdV_u3(u_new, x_new, t_new, eps = (torch.rand(()) - 0.5) * 2. * .4) # galileo
    
    return u_new, x_new, t_new
