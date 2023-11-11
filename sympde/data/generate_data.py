

import numpy as np
from math import pi
from tqdm import tqdm
from scipy.integrate import solve_ivp

class GeneratePDEData:
    """
    Adapted from Brandstetter, J., Welling, M., Worrall, D.E., 2022. Lie Point Symmetry Data Augmentation for Neural PDE Solvers. https://doi.org/10.48550/arXiv.2202.07643
        https://github.com/brandstetter-johannes/LPSDA/blob/master/notebooks/data_generation.ipynb
    """

    def __init__(self, L, T, Nx, Nt, tol = 1e-6):
        self.L = L
        self.Nx = Nx
        self.Nt = Nt
        self.x = np.linspace(0, (1-1.0/Nx)*L, Nx)
        self.t = np.linspace(0, T, Nt)
        self.tol = tol

        self.dx = L/Nx
        self.dt = T/(Nt-1)

    def generate_params(self) -> (int, np.ndarray, np.ndarray, np.ndarray):
        """
        Returns parameters for initial conditions.
        Args:
            None
        Returns:
            int: number of Fourier series terms
            np.ndarray: amplitude of different sine waves
            np.ndarray: phase shift of different sine waves
            np.ndarray: frequency of different sine waves
        """
        N = 10
        lmin, lmax = 1, 3
        A = (np.random.rand(1, N) - 0.5)
        phi = 2.0*np.pi*np.random.rand(1, N)
        l = np.random.randint(lmin, lmax, (1, N))
        return (N, A, phi, l)

    def get_init_cond(self, x: np.ndarray, L: int) -> np.ndarray:
        """
        Return initial conditions based on initial parameters.
        Args:
            x (np.ndarray): input array of spatial grid
            L (float): length of the spatial domain
            params (Optinal[list]): input parameters for generating initial conditions
        Returns:
            np.ndarray: initial condition
        """
        params = self.generate_params()
        N, A, phi, l = params   
        u0 = np.sum(A * np.sin((2 * np.pi * l * x[:, None] / L ) + phi), -1)
        return u0

    def solve_pde(self, pde_func):

        u0 = self.get_init_cond(self.x, self.L)
        t = self.t

        sol = solve_ivp(fun=pde_func, 
                    t_span=[t[0], t[-1]], 
                    y0=u0, 
                    method='Radau', 
                    t_eval=t, 
                    atol=self.tol, 
                    rtol=self.tol)
        
        return sol.y.T, (self.dx, self.dt)

    def generate_data(self, pde_func, N_samples: int = 1):
        us = np.full((N_samples, self.Nt, self.Nx), np.nan)


        for i in tqdm(range(N_samples), desc = f'Generating data pde_func!'):
            u, (dx, dt) = self.solve_pde(pde_func)
            u_tf = u.shape[0]
            if u_tf < self.Nt: 
                print(f'Warning: x_tf = {u_tf} < Nt = {self.Nt}')
            us[i, :u_tf, :] = u

        return us, dx, dt
    
