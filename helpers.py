import numpy as np

def normalize(data):
    max_value=63
    norm_data= -1 + 2 * np.array(data) / max_value
    return norm_data

def expand_channels(x):
    return x.expand(3, -1, -1)
