def get_dict_item(dict, idx = 0):
    """Return the idx-th item of a dictionary."""
    key, value = list(dict.items())[idx]
    return key, value

def read_lines(file):
    """Read a pickle object."""
    with open(file, 'r') as f:
        lines = f.readlines()
    return lines