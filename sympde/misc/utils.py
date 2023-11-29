import pickle

def get_dict_item(dict, idx = 0):
    """Return the idx-th item of a dictionary."""
    key, value = list(dict.items())[idx]
    return key, value

def read_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return lines

def write_lines(file, lines):
    with open(file, 'w') as f:
        f.writelines(lines)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)