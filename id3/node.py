class Node():
    """ A node class for used to build trees

    Parameters
    ----------
    values : list of values to be saved to be saved by the node
    is_feature : list of values to be saved to be saved by the node
    """
    def __init__(self, values, is_feature=True):
        self.values = values

    #def add_child
