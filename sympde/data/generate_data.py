

import numpy as np
from math import pi
from tqdm import tqdm
from scipy.integrate import solve_ivp

class GeneratePDEData:
    """
    Adapted from Brandstetter, J., Welling, M., Worrall, D.E., 2022. Lie Point Symmetry Data Augmentation for Neural PDE Solvers. https://doi.org/10.48550/arXiv.2202.07643
        https://github.com/brandstetter-johannes/LPSDA/blob/master/notebooks/data_generation.ipynb
    """

    def __init__(self, Lmax, Tmax, Nx, Nt, tol = 1e-6):
        self.Lmax = Lmax
        self.Tmax = Tmax
        self.Nx = Nx
        self.Nt = Nt
        self.tol = tol

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

        # l1, l2 = self.Lmax - self.Lmax/10, self.Lmax + self.Lmax/10
        # t1, t2 = self.Tmax - self.Tmax/10, self.Tmax + self.Tmax/10
        # L = np.random.uniform(l1, l2)
        # T = np.random.uniform(t1, t2)
        L = self.Lmax
        T = self.Tmax

        x = np.linspace(0, (1-1.0/self.Nx)*L, self.Nx)
        t = np.linspace(0, T, self.Nt)
        dx = L/self.Nx
        dt = T/(self.Nt-1)


        u0 = self.get_init_cond(x, L)

        sol = solve_ivp(fun=pde_func, 
                    t_span=[t[0], t[-1]], 
                    y0=u0, 
                    method='Radau', 
                    t_eval=t, 
                    atol=self.tol, 
                    rtol=self.tol)
        
        return sol.y.T, (dx, dt)

    def generate_data(self, pde_func, N_samples: int = 1, tqdm_desc: str = 'Generating data pde_func!'):
        us = np.full((N_samples, self.Nt, self.Nx), np.nan)
        dxs, dts = [], []


        for i in tqdm(range(N_samples), desc = tqdm_desc):
            u, (dx, dt) = self.solve_pde(pde_func)
            u_tf = u.shape[0]
            if u_tf < self.Nt: 
                print(f'Warning: x_tf = {u_tf} < Nt = {self.Nt}')
            us[i, :u_tf, :] = u
            dxs.append(dx)
            dts.append(dt)

        dxs = np.array(dxs)
        dts = np.array(dts)
        return us, dxs, dts
    
