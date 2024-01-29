import numpy as np
import os
import torch

class NumpyUtils:
    def __init__(self, dir):
        self.dir = dir
        self.listdir = os.listdir(self.dir)

    def load(self, filename):
        return np.load(os.path.join(self.dir, filename + '.npy'))
    
    def save(self, filename, array):
        np.save(os.path.join(self.dir, filename + '.npy'), array)

    def load_all(self):
        return {filename[:-4] : self.load(filename[:-4]) for filename in os.listdir(self.dir) if '.npy' in filename}
    
    def save_all(self, dict):
        for filename, array in dict.items():
            self.save(filename, array)

class Results:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def tensor_operation(x, operation):
    assert isinstance(x, np.ndarray)
    return operation(torch.from_numpy(x)).numpy()

def numpy_operation(x, operation):
    assert isinstance(x, torch.Tensor)
    return torch.from_numpy(operation(x.numpy()))