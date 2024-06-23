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

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def tensor_operation(x, operation):
    assert isinstance(x, np.ndarray)
    return operation(torch.from_numpy(x)).numpy()

def numpy_operation(x, operation):
    assert isinstance(x, torch.Tensor)
    return torch.from_numpy(operation(x.numpy()))

def print_nested_keys(d, indent=0):
    """
    Recursively prints all the keys of a nested dictionary.

    Parameters:
    d (dict): The nested dictionary whose keys are to be printed.
    indent (int): The current indentation level (used for pretty printing).
    """
    # Iterate through the dictionary items
    for key, value in d.items():
        # Print the current key with appropriate indentation
        print('  ' * indent + str(key))
        
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            print_nested_keys(value, indent + 1)
        elif isinstance(value, list):
            # If the value is a list, iterate through the list items
            for item in value:
                if isinstance(item, dict):
                    # If the item is a dictionary, recurse into it
                    print_nested_keys(item, indent + 1)