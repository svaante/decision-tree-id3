import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.externals import six
import namedtuple

CalcRecord = namedtuple('CalcRecord', ['feature_idx', 'entropy', 'info', 'pivot', 'type'])
NUM = 0
NOM = 1

class BaseSplitter(six.with_metaclass(ABCMeta)):

    def __init__(self, is_numerical):
        self._X = None
        self._y = None
        self.calc_info = None
        self.examples_idx = None


    def _info(self, x, y):
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


    def calc(self, X, y, examples_idx, features_idx):
        """
        Returns
        -------
        : int
            Feature index for split
        : dict
            Returns all information used for split
        """
        self.X_ = X[np.ix_(examples_idx, features_idx)]
        self.y_ = y[examples_idx]
        self.examples_idx = examples_idx


    def split(self, nominal_values=None):
        """
        Returns
        -------
        : list
            Array contaning the the subsets for the split
        """
        """
        if self._X is None:
            return None
        if nominal_values is None:
            feature_idx = self.calc_info["features_idx"]
            pivot = self.calc_info["pivot"]
            return [np.where(self.X_[:, feature_idx] < pivot),
                    np.where(pivot <= self.X_[:, feature_idx])]
        else:
            bags = [None] * len(nominal_values)
            for value, i in enumerate(nominal_values):
                bags[i] = np.where(self.X_[:, feature_idx] == value)
            return bags
        """
        pass


class NumericalSplitter(BaseSplitter):


    def calc(self, X, y, examples_idx, features_idx):
        super().calc()
        for i, x_ in zip(range(self.X_.shape[1]), self.X_.T):
            tmp_info, tmp_pivot = self._info(x_, self.y_)
            min_info, min_info_pivot, min_info_idx = self._update_info(min_info,
                                                                       tmp_info,
                                                                       min_info_idx,
                                                                       i
                                                                       )
        self.split_idx = features_id[min_info_idx]
        entropy = self._entropy(self.y_)
        self.calc_record = Point(features_idx[min_info_idx],
                                 entropy,
                                 min_info,
                                 min_info_pivot,
                                 NUM)
        return self.calc_record


    def _update_info(self, min_info, tmp_info, min_info_idx, tmp_idx):
        return (min_info, min_info_idx) if min_info[0] < tmp_info[0] else (tmp_info, tmp_idx)


    def _info(self, x, y):
        """ Info for numerical feature feature_values
        sort values then find the best split value

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
        : float
            pivot used set1 < pivot <= set2
        """
        sorted_idx = np.argsort(x, kind='quicksort')
        n = x.size
        sorted_y = y[sorted_idx]
        sorted_x = x[sorted_idx]
        min_info = float('inf')
        min_info_pivot = 0
        for i in range(n - 1):
            if sorted_y[i] != sorted_y[i + 1]:
                tmp_info = i * self._entropy(sorted_y[0: i]) + \
                           (n - i) * self._entropy[i:]
                if tmp_info < min_info:
                    min_info = tmp_info
                    min_info_pivot = sorted_x[i + 1]
        return min_info * np.true_divide(1, n), min_info_pivot


class NominalSplitter(BaseSplitter):

    def calc(self, X, y, examples_idx, features_idx):
        super.calc(X, y, examples_idx, features_idx)
        for i, x_ in zip(range(self.X_.shape[1]), self.X_.T):
            tmp_info = self._info(x_, self.y_)
            tmp_info, min_info_idx = self._update_info(min_info, tmp_info, min_info_idx, i)
        self.split_idx = features_idx[min_info_idx]
        entropy = self._entropy(self.y_)
        self.calc_record = Point(features_idx[min_info_idx],
                                 entropy,
                                 min_info,
                                 None,
                                 NOM)
        return self.calc_record


    def _info(self, x, y):
        """ Info for nominal feature feature_values
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


    def _update_info(self, min_info, tmp_info, min_info_idx, tmp_idx):
        return (min_info, min_info_idx) if min_info < tmp_info else (tmp_info, tmp_idx)


class HybridSplitter(BaseSplitter):

    def __init__(self, is_numerical):
        self._is_numerical = is_numerical
        self._numerical_splitter = NumericalSplitter()
        self._nominal_splitter = NominalSplitter()

    def calc(self, X, y, examples_idx, features_idx):
        super.calc(X, y, examples_idx, features_idx)
        min_info = float('inf')
        min_info_sub_idx = 0
        for feature, idx in enumerate(self._X.T):
            splitter = self._get_splitter(idx)
            tmp_info = splitter._info(feature, y)
            min_info, min_info_idx = splitter._update_info(min_info, tmp_info, min_info_idx, idx)
        self.split_idx = features_idx[min_info_sub_idx]
        entropy = self._entropy(self.y_)
        self.calc_record = CalcRecord(features_idx[min_info_sub_idx],
                                 entropy,
                                 min_info,
                                 None,
                                 NOM)
        return self.calc_record

    def _get_splitter(self, idx):
        if self._is_numerical[idx]:
            return self._numerical_splitter
        else:
            return self._nominal_splitter


