class Node():
    """ A node class for used to build trees

    Parameters
    ----------
    value : value of the node should be either class or feature value
    details : list of values to be saved to be saved by the node
    estimator : label estimator
    is_feature : if current node represent a feature or classification
    """
    def __init__(self,
                 value,
                 is_feature=False,
                 details={}):
        self.value = value
        self.is_feature = is_feature
        self.details = details
        self.children = list()

    def add_child(self, node, split_record):
        """ Add a child to node

        Parameters
        ----------
        node : child node to be added
        edge_value : attribute value for the edge
        is_feature : if current node represent a feature or classification
        """
        self.children.append((node, split_record))
