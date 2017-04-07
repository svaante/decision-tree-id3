class Node():
    """ A node class for used to build trees

    Parameters
    ----------
    name : name of the node should be either class or feature name
    details : list of values to be saved to be saved by the node
    estimator : label estimator
    is_feature : if current node represent a feature or classification
    """
    def __init__(self,
                 name,
                 estimator,
                 is_feature=True,
                 details=None):
        self.name = name
        self.estimator = estimator
        self.is_feature = is_feature
        self.details = details
        self.children = list()

    def add_child(self, node, edge_value):
        """ Add a child to node

        Parameters
        ----------
        node : child node to be added
        edge_value : attribute value for the edge
        is_feature : if current node represent a feature or classification
        """
        self.children.append((node, edge_value))

    def print_tree(self, prefix=""):
        print(prefix + str(self.name), end="") 
        print(" - Feature" if self.is_feature else " - Classification")
        for child, edge in self.children:
            print(prefix + edge.decode('UTF-8'))
            child.print_tree(prefix + "\t")
