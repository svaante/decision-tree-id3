import numpy as np


class SplitRecord():
    LESS = 0
    GREATER = 1

    def __init__(self, calc_record, bag, value_encoded, value_decoded):
        self.calc_record = calc_record
        self.bag = bag
        self.value_encoded = value_encoded
        self.value_decoded = value_decoded
        self.size = len(bag) if bag is not None else 0


class CalcRecord():
    NUM = 0
    NOM = 1

    def __init__(self,
                 split_type,
                 info,
                 feature_idx=None,
                 feature_name=None,
                 entropy=None,
                 pivot=None,
                 attribute_counts=None,
                 class_counts=None):
        self.split_type = split_type
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.entropy = entropy
        self.info = info
        self.pivot = pivot
        self.class_counts = class_counts
        self.attribute_counts = attribute_counts

    def __lt__(self, other):
        if not isinstance(other, CalcRecord):
            return True
        return self.info < other.info


class Splitter():

    def __init__(self, X, y, is_numerical, encoders, feature_names):
        self.X = X
        self.y = y
        self.is_numerical = is_numerical
        self.encoders = encoders
        self.feature_names = feature_names

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
        classes, count = np.unique(y, return_counts=True)
        p = np.true_divide(count, n)
        res = np.sum(np.multiply(p, np.log2(np.reciprocal(p))))
        if return_class_counts:
            return res, np.stack((classes, count), axis=-1)
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
        unique, count = np.unique(x, return_counts=True)
        for value, p in zip(unique, count):
            info += p * self._entropy(y[x == value])
        return CalcRecord(CalcRecord.NOM, info * np.true_divide(1, n),
                          attribute_counts=np.stack((unique, count), axis=-1))

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
        sorted_idx = np.argsort(x, kind='quicksort')
        n = x.size
        sorted_y = y[sorted_idx]
        sorted_x = x[sorted_idx]
        min_info = float('inf')
        min_info_pivot = 0
        min_attribute_counts = np.zeros((n, 2))
        for i in range(n - 1):
            if sorted_y[i] != sorted_y[i + 1]:
                tmp_info = (i + 1) * self._entropy(sorted_y[0: i]) + \
                           (n - (i + 1)) * self._entropy(sorted_y[i:])
                if tmp_info < min_info:
                    min_attribute_counts[SplitRecord.LESS, 1] = i + 1
                    min_attribute_counts[SplitRecord.GREATER, 1] = n - i + 1
                    min_info = tmp_info
                    min_info_pivot = sorted_x[i + 1]
        return CalcRecord(CalcRecord.NUM,
                          min_info * np.true_divide(1, n),
                          pivot=min_info_pivot,
                          attribute_counts=min_attribute_counts)

    def _split_nominal(self, X_, examples_idx, calc_record):
        ft_idx = calc_record.feature_idx
        values = self.encoders[ft_idx].encoded_classes_
        classes = self.encoders[ft_idx].classes_
        split_records = [None] * len(values)
        for val, i in enumerate(values):
            split_records[i] = SplitRecord(calc_record,
                                           examples_idx[X_[:, ft_idx] == val],
                                           val,
                                           classes[i])
        return split_records

    def _split_numerical(self, X_, examples_idx, calc_record):
        idx = calc_record.feature_idx
        split_records = [None] * 2
        split_records[0] = SplitRecord(calc_record,
                                       examples_idx[X_[:, idx]
                                                    < calc_record.pivot],
                                       SplitRecord.LESS,
                                       "<{}".format(calc_record.pivot))
        split_records[1] = SplitRecord(calc_record,
                                       examples_idx[X_[:, idx]
                                                    >= calc_record.pivot],
                                       SplitRecord.GREATER,
                                       ">={}".format(calc_record.pivot))
        return split_records

    def calc(self, examples_idx, features_idx):
        X_ = self.X[np.ix_(examples_idx, features_idx)]
        y_ = self.y[examples_idx]
        calc_record = None
        for idx, feature in enumerate(X_.T):
            tmp_calc_record = None
            if self.is_numerical[features_idx[idx]]:
                tmp_calc_record = self._info_numerical(feature, y_)
            else:
                tmp_calc_record = self._info_nominal(feature, y_)
            if tmp_calc_record < calc_record:
                ft_idx = features_idx[idx]
                calc_record = tmp_calc_record
                calc_record.feature_idx = ft_idx
                if self.feature_names is not None:
                    calc_record.feature_name = self.feature_names[ft_idx]
                else:
                    calc_record.feature_name = str(ft_idx)
        calc_record.entropy, calc_record.class_counts = self._entropy(y_, True)
        return calc_record

    def split(self, examples_idx, calc_record):
        X_ = self.X[np.ix_(examples_idx)]
        if self.is_numerical[calc_record.feature_idx]:
            return self._split_numerical(X_, examples_idx, calc_record)
        else:
            return self._split_nominal(X_, examples_idx, calc_record)
