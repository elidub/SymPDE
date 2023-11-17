import pickle
import torch

from data.lpda_data_aug import to_coords

def d_to_LT(us, dx, dt):
    Nt, Nx = us[0].shape if len(us.shape) == 3 else us.shape
    L = dx * Nx
    T = dt * (Nt - 1)

    # check if L is float
    if isinstance(L, float):
        return L, T
    
    assert (L == L[0]).all() and (T == T[0]).all()
    return int(L[0]), int(T[0])

def d_to_coords(u, dx, dt):
    Nt, Nx = u.shape
    x = torch.arange(0, Nx) * dx
    t = torch.arange(0, Nt) * dt
    X = to_coords(x, t)
    return X


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)