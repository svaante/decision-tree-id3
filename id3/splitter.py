import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.extarnals import six


class Splitter(six.with_metaclass(ABCMeta)):

    def __init__(self):
        pass

    def _entropy(self, y):
        """ Entropy for the classes in the array y
        :math: \sum_{x \in X} p(x) \log_{2}(1/p(x)) :math: from
        https://en.wikipedia.org/wiki/ID3_algorithm

        Parameters
        ----------
        y : nparray of shape [n remaining attributes]
            containing the class names

        Returns
        -------
        : float
            information for remaining examples given feature
        """
        n = y.shape[0]
        if n <= 0:
            return 0
        _, count = np.unique(y, return_counts=True)
        p = np.true_divide(count, n)
        return np.sum(np.multiply(p, np.log2(np.reciprocal(p))))

    @abstractmethod
    def calc(X, y, examples_idx, features_idx, gain_ratio=False):
        """
        Returns
        -------
        : dict
            Returns all used information used for split
        : int
            Feature index for split
        """
        pass

    @abstractmethod
    def split(values):
        """
        Returns
        -------
        : np.array [n subsets]
            Array contaning the the subsets for the split
        """
        pass

class NominalSplitter(Splitter):

    def __init__(self):
        self._X = None
        self._y = None

    def _info(self, x, y):
        """ info for feature feature_values
        :math: p(a)H(a) :math: from
        https://en.wikipedia.org/wiki/ID3_algorithm

        Parameters
        ----------
        x : np.array of shape [n remaining examples]
            containing feature values
        y : np.array of shape [n remaining examples]
            containing relevent class

        Returns
        -------
        : float
            information for remaining examples given feature
        """
        info = 0
        n = x.shape[0]
        unique, count = np.unique(x, return_counts=True)
        for value, p in zip(unique, count):
            info += p * self._entropy(y[x == value])
        return info * np.true_divide(1, n)

    def calc(self, X, y, examples_idx, features_idx, gain_ratio=False):
        """
        Returns
        -------
        : int
            Feature index for split
        : dict
            Returns all used information used for split
        """
        calc_info = {
                     'entropy': None,
                     'info': None,
                     'split': None,
                     'info_split': None
                    }
        self.X_ = X[np.ix_(examples_idx, features_idx)]
        self.y_ = y[examples_idx]
        features_info = np.apply_along_axis(self._info, 0, self.X_, self.y_)
        argmin_features_info = np.argmin(features_info)
        entropy = self.super()._entropy(self.y_)
        info = features_info[argmin_features_info]
        calc_info['entropy'] = entropy
        calc_info['loss'] = loss
        return features_idx[argmin_features_info], calc_info


    def split(values):
        """
        Returns
        -------
        : np.array [n subsets]
            Array contaning the the subsets for the split
        """
        pass

class NumericalSplitter(Splitter):


