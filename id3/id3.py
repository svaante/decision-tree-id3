"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import numbers
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .builder import TreeBuilder, Tree
from .splitter import Splitter, SplitRecord, CalcRecord
from .utils import check_numerical_array, ExtendedLabelEncoder


# TODO(svaante): Intrinsic information
# http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf
class Id3Estimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation .

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, max_depth=None, min_samples_split=2, prune=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.prune = prune

    def fit(self, X, y, feature_names=None, check_input=True):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Attributes
        ----------
        n_features: int
            The number of features when ``fit`` is performed.

        X_encoders : list
            List of LabelEncoders that transforms input from labels to binary
            encodings and vice versa.

        y_encoder : LabelEncoder
            LabelEncoders that transforms output from labels to binary
            encodings and vice versa.

        Returns
        -------
        self : object
            Returns self.
        """
        X_, y = check_X_y(X, y)
        self.feature_names = feature_names
        if self.prune:
            X_, X_test, y, y_test = train_test_split(X_, y, test_size=0.2)

        max_np_int = np.iinfo(np.int32).max
        if not isinstance(self.max_depth, (numbers.Integral, np.integer)):
            max_depth = max_np_int
        else:
            max_depth = self.max_depth

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            min_samples_split = (1 if self.min_samples_split < 1
                                 else self.min_samples_split)
        else:
            min_samples_split = 1

        n_samples, self.n_features = X_.shape
        self.is_numerical = [False] * self.n_features
        if (self.feature_names is not None and not
                self.n_features <=
                len(self.feature_names) <=
                (self.n_features + 1)):
            raise ValueError(("feature_names needs to have the same "
                              "number of elements as features in X"),)
        self.X = np.zeros(X_.shape, dtype=np.float32)
        self.X_encoders = [ExtendedLabelEncoder() for _ in
                           range(self.n_features)]
        for i in range(self.n_features):
            if check_input and check_numerical_array(X_[:, i]):
                self.is_numerical[i] = True
                self.X[:, i] = X_[:, i]
            else:
                self.X[:, i] = self.X_encoders[i].fit_transform(X_[:, i])
        self.y_encoder = ExtendedLabelEncoder()
        self.y = self.y_encoder.fit_transform(y)
        splitter_ = Splitter(self.X,
                             self.y,
                             self.is_numerical,
                             self.X_encoders,
                             self.feature_names)
        self.builder_ = TreeBuilder(splitter_,
                                    self.y_encoder,
                                    n_samples,
                                    self.n_features,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    prune=self.prune)
        self.tree_ = Tree()
        self.builder_.build(self.tree_, self.X, self.y)

        if self.prune:
            self._reduced_error_pruning(X_test, y_test)

        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features_idx]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        check_is_fitted(self, 'tree_')
        X = check_array(X)
        X_ = np.zeros(X.shape)
        ret = np.empty(X.shape[0], dtype=X.dtype)
        for i in range(self.n_features):
            if self.is_numerical[i]:
                X_[:, i] = X[:, i]
            else:
                X_[:, i] = self.X_encoders[i].transform(X[:, i])
        for i, x in enumerate(X_):
            node = self.tree_.root
            while(node.is_feature):
                value = x[node.details.feature_idx]
                for child, split_record in node.children:
                    if split_record.calc_record.split_type == CalcRecord.NOM:
                        if split_record.value_encoded == value:
                            node = child
                            break
                    elif split_record.calc_record.split_type == CalcRecord.NUM:
                        if (split_record.value_encoded ==
                            SplitRecord.GREATER and
                                value >= split_record.calc_record.pivot):
                            node = child
                            break
                        elif (split_record.value_encoded ==
                              SplitRecord.LESS and
                              value < split_record.calc_record.pivot):
                            node = child
                            break
            ret[i] = node.value
        return ret

    def _reduced_error_pruning(self, X_test, y_test):
        y_pred = self.predict(X_test)
        base_score = accuracy_score(y_pred, y_test)
        for node in self.tree_.feature_nodes:
            if not node.is_feature:
                continue
            encoded_class = node.details.class_counts[0, 0]
            decoded_class = self.y_encoder.inverse_transform(encoded_class)
            tmp_value = node.value
            node.value = decoded_class
            node.is_feature = False
            y_pred = self.predict(X_test)
            new_score = accuracy_score(y_pred, y_test)
            if new_score < base_score:
                node.value = tmp_value
                node.is_feature = True
            else:
                node.children = []
