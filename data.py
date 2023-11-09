

import numpy as np
from math import pi
from tqdm import tqdm
from scipy.integrate import solve_ivp

from pde import PDE, CartesianGrid, ScalarField, MemoryStorage

def generate_params(N_params: int = 10, seed: int = 42):
    np.random.seed(seed)
    As = np.random.uniform(-0.5, 0.5, N_params)
    ws = np.random.choice([1, 2, 3], N_params)
    phis = np.random.uniform(0, 2*pi, N_params)

    return (As, ws, phis)


# class CreatePDEData_SymPDE:
#     def __init__(self, pde_exp: str = 'd2_dx2(u)', Nt: int = 600):
#         self.pde_exp = pde_exp
#         self.Nt = Nt
#         self.grid_sizes = [50]
#         self.N_init_cond = 10

#     def get_init_cond(self, N: int) -> str:
#         """
#         Sample initial condition as a sum of N sine functions
#         """
#         params = generate_params()
#         sins = [f'{A} * sin({w} * x + {phi})' for A, w, phi in zip(*params)]
#         init_cond = ' + '.join(sins)
#         return init_cond


#     def solve_pde(self, 
#             pde_exp, dt, Nt,
#             bounds = [[0, 2 * pi]], grid_sizes = [50]
#         ):

#         init_cond = self.get_init_cond(self.N_init_cond)

#         t_range = 1
#         dt = t_range/(Nt - 1)
#         dt_track = t_range/(Nt - 1)

#         eq = PDE({"u": pde_exp})
#         grid = CartesianGrid(bounds, grid_sizes, periodic=True)
#         state = ScalarField.from_expression(grid, init_cond)

#         # solve the equation and store the trajectory
#         storage = MemoryStorage()
#         eq.solve(state, 
#                 t_range=t_range, 
#                 dt = dt, 
#                 adaptive = True,
#                 tracker=[storage.tracker(dt_track)],
#         )

#         return storage
    
#     def get_data(self, N_samples: int = 1):
#         data = np.zeros((N_samples, self.Nt, *self.grid_sizes))

#         for i in tqdm(range(N_samples), desc = f'Generating data for {self.pde_exp}!'):
#             storage = self.solve_pde(self.pde_exp, self.dt, self.Nt)
#             data[i] = storage.data

#         storage.data = np.zeros_like(storage.data)

#         return data, storage
    
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
        u = np.sum(A * np.sin((2 * np.pi * l * x[:, None] / L ) + phi), -1)
        return u

    def solve_pde(self, pde_func, tol = 1e-6):

        u0 = self.get_init_cond(self.x, self.L)
        t = self.t

        sol = solve_ivp(fun=pde_func, 
                    t_span=[t[0], t[-1]], 
                    y0=u0, 
                    method='Radau', 
                    t_eval=t, 
                    # args=(L,), 
                    atol=tol, 
                    rtol=tol)
        
        ts = sol.y.T[::-1]
        
        return ts

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

        for i in tqdm(range(N_samples), desc = f'Generating data pde_func!'):
            ts = self.solve_pde(pde_func, tol)
            data[i] = ts

        return data