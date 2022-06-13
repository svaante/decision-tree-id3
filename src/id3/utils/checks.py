import numpy as np


def check_numerical_array(array):
    """ Check if all values in a 1d array are numerical. Raises error if array
        is more than 1d.

    Parameters
    ----------
    array : nparray
        containing the class names

    Returns
    -------
    result : bool
        True if all values in array are numerical, otherwise false
    """
    try:
        if array.ndim > 1:
            raise ArithmeticError("Found array with dim {}. Expected = 1."
                                  .format(array.ndim))
        array.astype(np.float64)
        return True
    except ValueError:
        return False
