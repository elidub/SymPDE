import numpy as np

def write_lines(file, lines):
    with open(file, 'w') as f:
        f.writelines(lines)

def write_line(file, line):
    with open(file, 'w') as f:
        f.write(line)

def read_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return lines

def clean_val(val):
    val = str(val)
    # print(val)
    if val == 'nan': val = np.nan
    elif ',' in val: 
        val = val[1:-1].split(', ')
    else:
        val = [val]
    return val

def listify(comb_dict):
    for key, value in comb_dict.items():
        if not isinstance(value, list):
            comb_dict[key] = [value]
    return comb_dict

def generate_combinations(dictionary):
    # Base case: if the dictionary is empty, return an empty list
    if not dictionary:
        return [""]

    # Take one key-value pair from the dictionary
    key, values = next(iter(dictionary.items()))
    remaining_dict = dictionary.copy()
    del remaining_dict[key]

    # Recursively generate combinations for the remaining dictionary
    combinations = generate_combinations(remaining_dict)

    # Generate all possible combinations with the current key-value pair
    output_lines = []

    # if key == 'implicit_layer_dims':
        # print(values)

    for value in values:
        if key in ['implicit_layer_dims', 'grid_sizes']:
            print(key, value)
            value = f'"{value}"'
        for combination in combinations:
            if value == 'True':
                value = ''
            if value is None:
                line = combination
            elif combination == "":
                # if key in ['implicit_layer_dims', 'grid_sizes']:
                    # value = f'"{value}"'
                line = f"--{key} {value}" 
            else:
                if isinstance(value, list):
                    value = " ".join([str(v) for v in value])
                line = f"{combination} --{key} {value}"
            output_lines.append(line)

    return output_lines

def dict_to_array(comb_dict):
    # comb_dict = listify(comb_dict)
    output_lines = generate_combinations(comb_dict)
    output_lines = "\n".join(output_lines)
    return output_lines