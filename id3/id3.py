"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import numbers
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .node import Node
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
    def __init__(self, max_depth=None, min_samples_split=2, pruner=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.pruner = pruner

    def _build(self, examples_idx, features_idx, depth=0):
        """ Builds the tree with the self.X data and self.y classes

        Parameters
        ----------
        examples_idx: np.array
            data row(s) to be considered
        features_idx: np.array
            feature colum(s) to be considered

        Returns
        -------
        root : Node
            root node of tree
        """
        unique, counts = np.unique(self.y[examples_idx], return_counts=True)
        classification = unique[np.argmax(counts)]
        classification_name = self.y_encoder.inverse_transform(classification)
        if features_idx.size == 0 or unique.size == 1:
            node = Node(classification_name)
            self.classification_nodes.append(node)
            return node

        calc_record = self.splitter_.calc(examples_idx, features_idx)
        split_records = self.splitter_.split(examples_idx, calc_record)
        new_features_idx = np.delete(features_idx,
                                     np.where(features_idx ==
                                              calc_record.feature_idx))
        root = Node(calc_record.feature_name,
                    is_feature=True,
                    details=calc_record)
        self.feature_nodes.append(root)
        for record in split_records:
            if record.size == 0:
                node = Node(classification_name)
                self.classification_nodes.append(node)
                root.add_child(node, record)
            else:
                root.add_child(self._build(record.bag,
                               new_features_idx, depth+1),
                               record)
        return root

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
        self.feature_nodes = []
        self.classification_nodes = []
        X_, y = check_X_y(X, y)
        prune = self.pruner == 'ReducedError'
        if prune:
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

        n_samples, self.n_features_idx = X_.shape
        self.is_numerical = [False] * self.n_features_idx
        if (self.feature_names is not None and not
                self.n_features_idx <=
                len(self.feature_names) <=
                (self.n_features_idx + 1)):
            raise ValueError(("feature_names needs to have the same "
                              "number of elements as features in X"),)
        self.X = np.zeros(X_.shape, dtype=np.float32)
        self.X_encoders = [ExtendedLabelEncoder() for _ in
                           range(self.n_features_idx)]
        for i in range(self.n_features_idx):
            if check_input and check_numerical_array(X_[:, i]):
                self.is_numerical[i] = True
                self.X[:, i] = X_[:, i]
            else:
                self.X[:, i] = self.X_encoders[i].fit_transform(X_[:, i])
        self.y_encoder = ExtendedLabelEncoder()
        self.y = self.y_encoder.fit_transform(y)
        self.splitter_ = Splitter(self.X,
                                  self.y,
                                  self.is_numerical,
                                  self.X_encoders,
                                  self.feature_names)
        self.tree_ = self._build(np.arange(n_samples),
                                 np.arange(self.n_features_idx))

        if prune:
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
        for i in range(self.n_features_idx):
            if self.is_numerical[i]:
                X_[:, i] = X[:, i]
            else:
                X_[:, i] = self.X_encoders[i].transform(X[:, i])
        for i, x in enumerate(X_):
            node = self.tree_
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
        for node in self.feature_nodes:
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
