def get_dict_item(dict, idx = 0):
    """Return the idx-th item of a dictionary."""
    key, value = list(dict.items())[idx]
    return key, value