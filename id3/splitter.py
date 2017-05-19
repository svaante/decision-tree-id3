import numpy as np
from .utils import unique


class SplitRecord():
    LESS = 0
    GREATER = 1

    def __init__(self, calc_record, bag, value_encoded):
        self.calc_record = calc_record
        self.bag = bag
        self.value_encoded = value_encoded
        self.size = len(bag) if bag is not None else 0


class CalcRecord():
    NUM = 0
    NOM = 1

    def __init__(self,
                 split_type,
                 info,
                 feature_idx=None,
                 entropy=None,
                 pivot=None,
                 attribute_counts=None,
                 class_counts=None,
                 gain_ratio=None,
                 alive_features=None):
        self.split_type = split_type
        self.info = info
        self.feature_idx = feature_idx
        self.entropy = entropy
        self.pivot = pivot
        self.class_counts = class_counts
        self.attribute_counts = attribute_counts
        self.gain_ratio = gain_ratio
        self.alive_features = alive_features

    def __lt__(self, other):
        if not isinstance(other, CalcRecord):
            return True
        return self.info < other.info


class Splitter():

    def __init__(self, X, y, is_numerical, encoders, gain_ratio=False):
        self.X = X
        self.y = y
        self.is_numerical = is_numerical
        self.encoders = encoders
        self.gain_ratio = gain_ratio

    def _entropy(self, y, return_class_counts=False):
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
        classes, count = unique(y)
        p = np.true_divide(count, n)
        res = np.abs(np.sum(np.multiply(p, np.log2(p))))
        if return_class_counts:
            return res, np.vstack((classes, count)).T
        else:
            return res

    def _info_nominal(self, x, y):
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
        items, count = unique(x)
        for value, p in zip(items, count):
            info += p * self._entropy(y[x == value])
        return CalcRecord(CalcRecord.NOM,
                          info * np.true_divide(1, n),
                          attribute_counts=count)

    def _info_numerical(self, x, y):
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
        n = x.size
        sorted_idx = np.argsort(x, kind='quicksort')
        sorted_y = np.take(y, sorted_idx, axis=0)
        sorted_x = np.take(x, sorted_idx, axis=0)
        min_info = float('inf')
        min_info_pivot = 0
        min_attribute_counts = np.empty(2)
        for i in range(1, n):
            if sorted_x[i - 1] != sorted_x[i]:
                tmp_info = i * self._entropy(sorted_y[0: i]) + \
                           (n - i) * self._entropy(sorted_y[i:])
                if tmp_info < min_info:
                    min_attribute_counts[SplitRecord.LESS] = i
                    min_attribute_counts[SplitRecord.GREATER] = n - i
                    min_info = tmp_info
                    min_info_pivot = (sorted_x[i - 1] + sorted_x[i]) / 2.0
        return CalcRecord(CalcRecord.NUM,
                          min_info * np.true_divide(1, n),
                          pivot=min_info_pivot,
                          attribute_counts=min_attribute_counts)

    def _split_nominal(self, X_, examples_idx, calc_record):
        ft_idx = calc_record.feature_idx
        values = self.encoders[ft_idx].encoded_classes_
        split_records = [None] * len(values)
        for val, i in enumerate(values):
            split_records[i] = SplitRecord(calc_record,
                                           examples_idx[X_[:, ft_idx]
                                                        == val],
                                           val)
        return split_records

    def _split_numerical(self, X_, examples_idx, calc_record):
        idx = calc_record.feature_idx
        split_records = [None] * 2
        split_records[0] = SplitRecord(calc_record,
                                       examples_idx[X_[:, idx]
                                                    <= calc_record.pivot],
                                       SplitRecord.LESS)
        split_records[1] = SplitRecord(calc_record,
                                       examples_idx[X_[:, idx]
                                                    > calc_record.pivot],
                                       SplitRecord.GREATER)
        return split_records

    def _intrinsic_value(self, calc_record):
        """ Calculates the gain ratio using CalcRecord
        :math: - \sum_{i} \fraq{|S_i|}{|S|}\log_2 (\fraq{|S_i|}{|S|}):math:

        Parameters
        ----------
        calc_record : CalcRecord

        Returns
        -------
        : float
        """
        counts = calc_record.attribute_counts
        s = np.true_divide(counts, np.sum(counts))
        return np.abs(np.sum(np.multiply(s, np.log2(s))))

    def _gain_ratio(self, calc_record):
        return np.true_divide(calc_record.entropy - calc_record.info,
                              self._intrinsic_value(calc_record))

    def _is_close(self, a, b):
        return np.abs(a + b) <= (1e-08 + 1e-05 * np.abs(b))

    def _is_better(self, calc_record1, calc_record2):
        """Compares CalcRecords using gain ratio if present otherwise
           using the  information.

        Parameters
        ----------
        calc_record1 : CalcRecord
        calc_record2 : CalcRecord

        Returns
        -------
        : bool
            if calc_record1 < calc_record2
        """
        if calc_record1 is None:
            return True
        if calc_record2 is None:
            return False
        if self.gain_ratio:
            if calc_record1.gain_ratio is None:
                calc_record1.gain_ratio = self._gain_ratio(calc_record1)
            if calc_record2.gain_ratio is None:
                calc_record2.gain_ratio = self._gain_ratio(calc_record2)
            if self._is_close(calc_record1.gain_ratio,
                              calc_record2.gain_ratio):
                return calc_record1.info > calc_record2.info
            else:
                return calc_record1.gain_ratio < calc_record2.gain_ratio
        else:
            return calc_record1.info > calc_record2.info

    def calc(self, examples_idx, features_idx):
        """ Calculates information regarding optimal split based on
        information gain

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
        X_ = self.X[np.ix_(examples_idx, features_idx)]
        y_ = self.y[examples_idx]
        calc_record = None
        alive_features = [True] * features_idx.shape[0]
        entropy, class_counts = self._entropy(y_, True)
        for idx, feature in enumerate(X_.T):
            if np.max(feature) == np.min(feature):
                alive_features[idx] = False
                continue
            tmp_calc_record = None
            if self.is_numerical[features_idx[idx]]:
                tmp_calc_record = self._info_numerical(feature, y_)
            else:
                tmp_calc_record = self._info_nominal(feature, y_)
            tmp_calc_record.entropy = entropy
            tmp_calc_record.class_counts = class_counts
            if self._is_better(calc_record, tmp_calc_record):
                calc_record = tmp_calc_record
                calc_record.feature_idx = features_idx[idx]
                calc_record.alive_features = alive_features
        return calc_record

    def split(self, examples_idx, calc_record):
        X_ = self.X[np.ix_(examples_idx)]
        if self.is_numerical[calc_record.feature_idx]:
            return self._split_numerical(X_, examples_idx, calc_record)
        else:
            return self._split_nominal(X_, examples_idx, calc_record)
