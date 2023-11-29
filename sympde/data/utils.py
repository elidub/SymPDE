import pickle
import torch

def d_to_LT(us, dx, dt):
    Nt, Nx = us[0].shape if len(us.shape) == 3 else us.shape
    L = dx * Nx
    T = dt * (Nt - 1)

    return L, T

def to_coords(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Transforms the coordinates to a tensor X of shape [time, space, 2].
    Args:
        x: spatial coordinates
        t: temporal coordinates
    Returns:
        torch.Tensor: X[..., 0] is the space coordinate (in 2D)
                      X[..., 1] is the time coordinate (in 2D)
    """
    x_, t_ = torch.meshgrid(x, t)
    x_, t_ = x_.T, t_.T
    return torch.stack((x_, t_), -1)

def d_to_coords(u, dx, dt):
    Nt, Nx = u.shape
    x = torch.arange(0, Nx) * dx
    t = torch.arange(0, Nt) * dt
    X = to_coords(x, t)
    return X