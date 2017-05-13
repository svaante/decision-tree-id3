import numpy as np


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
                 item_count=None,
                 is_feature=False,
                 details={},
                 counts=None):
        self.value = value
        self.is_feature = is_feature
        self.details = details
        self.counts = counts
        self.children = list()
        self.predicts = None
        self.n_correct_predicts = 0
        self.item_count = item_count

    def add_predict_result(self, val, n_classes):
        if self.predicts is None:
            self.predicts = np.zeros(n_classes)
        self.predicts[val] += 1
        if self.value == val:
            self.n_correct_predicts += 1

    def add_child(self, node, split_record):
        """ Add a child to node

        Parameters
        ----------
        node : child node to be added
        edge_value : attribute value for the edge
        is_feature : if current node represent a feature or classification
        """
        self.children.append((node, split_record))
