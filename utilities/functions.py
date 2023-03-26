import pandas as pd

def readData(filename):
    df = pd.read_csv(filename)
    return df


def AttrValueNames(attrs, values_map, encoded_values:tuple):
    decoden_values = []
    if len(attrs) != len(encoded_values):
        raise Exception("Number of attributes and values must be equal")
    for idx, val in enumerate(encoded_values):
        attr_map = values_map[attrs[idx]]
        decoden_values.append(attr_map[val])
    
    return tuple(decoden_values)

