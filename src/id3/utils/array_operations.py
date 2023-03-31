import numpy as np


def unique(array):
    if getattr(array, "dtype", None) == np.float32:
        array = array.astype(int)
    counts = np.bincount(array)
    mask = counts != 0
    return np.nonzero(mask)[0], counts[mask]
