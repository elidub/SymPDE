import numpy as np
import torch

def fourier_shift(u: torch.Tensor, eps: float=0., dim: int=-1, order: int=0) -> torch.Tensor:
    """
    Shift in Fourier space.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter
        dim (int): dimension which is used for shifting
        order (int): derivative order
    Returns:
        torch.Tensor: Fourier shifted input
    """
    assert dim < 0
    n = u.shape[dim]
    u_hat = torch.fft.rfft(u, dim=dim, norm='ortho')
    # Fourier modes
    omega = torch.arange(n // 2 + 1)
    if n % 2 == 0:
        omega[-1] *= 0
    # Applying Fourier shift according to shift theorem
    fs = torch.exp(- 2 * np.pi * 1j * omega * eps)
    # For order>0 derivative is taken
    fs = (- 2 * np.pi * 1j * omega) ** order * fs
    for _ in range(-dim - 1):
        fs = fs[..., None]
    return torch.fft.irfft(fs * u_hat, n=n, dim=dim, norm='ortho')

def linear_shift(u: torch.Tensor, eps: float=0., dim:int=-1) -> torch.Tensor:
    """
    Linear shift.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter
        dim (int): dimension which is used for shifting
    Returns:
        Linear shifted input
    """
    n = u.shape[dim]
    # Shift to the left and to the right and interpolate linearly
    q, r = torch.div(eps*n, 1, rounding_mode='floor'), (eps * n) % 1
    q_left, q_right = q/n, (q+1)/n
    u_left = fourier_shift(u, eps=q_left, dim=-1)
    u_right = fourier_shift(u, eps=q_right, dim=-1)
    return (1-r) * u_left + r * u_right



