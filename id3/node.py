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
                 estimator,
                 name=None,
                 is_feature=False,
                 details={}):
        self.value = value
        self.estimator = estimator
        self.is_feature = is_feature
        self.details = details
        self.name = name
        self.children = list()

    def add_child(self, node, split_record):
        """ Add a child to node

        Parameters
        ----------
        node : child node to be added
        edge_value : attribute value for the edge
        is_feature : if current node represent a feature or classification
        """
        self.children.append((node, split_record.attribute_value))

    def print_tree(self, prefix=""):
        if self.is_feature:
            print(prefix + self.value, end="") 
            try:
                print(" - Entropy - {}".format(self.details.entropy), end="")
                print(" - Gain - {}".format(self.details.info), end="")
            except KeyError:
                pass
        else:
            print(prefix + str(self.value), end="")
        print(" - Feature" if self.is_feature else " - Classification")
        for child, edge in self.children:
            print(prefix + str(edge))
            child.print_tree(prefix + "\t")
