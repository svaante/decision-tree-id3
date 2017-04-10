def check_numerical_column(x):
    """ Check if all values in a column are numerical.

    Parameters
    ----------
    y : nparray of shape [n remaining attributes]
        containing the class names

    Returns
    -------
    : float
        information for remaining examples given feature
    """
    try:
        for val in x:
            float(val)
            print("Val: {}, float: {}".format(val, float(val)))
        return True
    except ValueError:
        return False
