import numpy as np


def unique(array):
    uni, inv = np.unique(array, return_inverse=True)
    return uni, np.bincount(inv)
