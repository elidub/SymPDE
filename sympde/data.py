

import numpy as np
from math import pi
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

from pde import PDE, CartesianGrid, ScalarField, MemoryStorage

# def generate_params(N_params: int = 10, seed: int = 42):
#     np.random.seed(seed)
#     As = np.random.uniform(-0.5, 0.5, N_params)
#     ws = np.random.choice([1, 2, 3], N_params)
#     phis = np.random.uniform(0, 2*pi, N_params)

#     return (As, ws, phis)
    
class CreatePDEData:
    def __init__(self, L, N, T, Nt):
        self.L = L
        self.N = N
        self.Nt = Nt
        self.x = np.linspace(0, (1-1.0/N)*L, N)
        self.t = np.linspace(0, T, Nt)

    # def generate_params(elf, N_params: int = 10, seed: int = 42):
    #     np.random.seed(seed)
    #     As = np.random.uniform(-0.5, 0.5, N_params)
    #     ws = np.random.choice([1, 2, 3], N_params)
    #     phis = np.random.uniform(0, 2*pi, N_params)

    #     return (As, ws, phis)


    # def get_init_cond(self, x: np.ndarray, L: int) -> np.ndarray:
    #     """
    #     Sample initial condition as a sum of N sine functions
    #     """
    #     A, w, phi = generate_params()   
    #     u = np.sum(A * np.sin(w * x[:, None] / L  + phi), -1)
    #     return u

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

    def solve_pde(self, pde_func, t = None, u0 = None, tol = 1e-6):

        if u0 is None:
            u0 = self.get_init_cond(self.x, self.L)
        if t is None:
            t = self.t

        sol = solve_ivp(fun=pde_func, 
                    t_span=[t[0], t[-1]], 
                    y0=u0, 
                    method='Radau', 
                    t_eval=t, 
                    # args=(L,), 
                    atol=tol, 
                    rtol=tol)
        
        return sol.y.T[::-1]

        # init_cond = self.get_init_cond(self.N_init_cond)

        # t_range = 1
        # dt = t_range/(Nt - 1)
        # dt_track = t_range/(Nt - 1)

        # assert dt_track <= dt

        # eq = PDE({"u": pde_exp})
        # grid = CartesianGrid(bounds, grid_sizes, periodic=True)
        # state = ScalarField.from_expression(grid, init_cond)

        # # solve the equation and store the trajectory
        # storage = MemoryStorage()
        # eq.solve(state, 
        #         t_range=t_range, 
        #         dt = dt, 
        #         adaptive = True,
        #         tracker=[storage.tracker(dt_track)],
        # )

        # return storage
    
    def get_data(self, pde_func, tol = 1e-6, N_samples: int = 1):
        data = np.zeros((N_samples, self.Nt, self.N))

        fail_count = 0

        for i in tqdm(range(N_samples), desc = f'Generating data pde_func!'):
            x = self.solve_pde(pde_func, tol = tol)
            x_tf = x.shape[0]
            if x_tf < self.Nt: 
                # print(f'Warning: x_tf = {x_tf} < Nt = {self.Nt}')
                fail_count += 1
            data[i, :x_tf, :] = x

        return data, fail_count
    
class PDE_Pseudospectral:
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