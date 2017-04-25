import numpy as np


def unique(array):
    counts = np.bincount(array)
    mask = counts != 0
    return np.nonzero(mask)[0], counts[mask]
