import pickle

def d_to_LT(us, dx, dt):
    Nt, Nx = us[0].shape if len(us.shape) == 3 else us.shape
    L = dx * Nx
    T = dt * (Nt - 1)
    return L, T

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)