"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split

from .node import Node
from .splitter import Splitter, SplitRecord, CalcRecord
from .pruner import BasePruner, ErrorPruner, CostPruner
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
    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def _build(self, examples_idx, features_idx):
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
            return Node(classification_name)

        calc_record = self._splitter.calc(examples_idx, features_idx)
        split_records = self._splitter.split(examples_idx, calc_record)
        new_features_idx = np.delete(features_idx,
                                     np.where(features_idx == calc_record.feature_idx))
        root = Node(calc_record.feature_name,
                    is_feature=True,
                    details=calc_record)
        for record in split_records:
            if record.size == 0:
                classification = self.y_encoder.inverse_transform(unique[np.argmax(counts)])
                root.add_child(Node(classification_name), record)
            else:
                root.add_child(self._build(record.bag,
                               new_features_idx),
                               record)
        return root

    def fit(self, X, y, feature_names=None, check_input=True, pruner=None):
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
        self.feature_names = feature_names
        self.pruner = pruner
        X_, y = check_X_y(X, y)
        prune = isinstance(self.pruner, BasePruner)
        if prune:
            X_, X_test, y, y_test = train_test_split(X_, y, test_size=0.2)

        n_samples, self.n_features_idx = X.shape
        is_numerical = [False] * self.n_features_idx
        if (self.feature_names is not None and not
                self.n_features_idx <= len(self.feature_names) <= (self.n_features_idx + 1)):
            raise ValueError(("feature_names needs to have the same "
                              "number of elements as features in X"),)
        self.X = np.zeros(X.shape, dtype=np.int)
        self.X_encoders = [ExtendedLabelEncoder() for _ in
                           range(self.n_features_idx)]
        for i in range(self.n_features_idx):
            if check_input and check_numerical_array(X_[:, i]):
                is_numerical[i] = True
                self.X[:, i] = X_[:, i]
            else:
                self.X[:, i] = self.X_encoders[i].fit_transform(X_[:, i])
        self.y_encoder = ExtendedLabelEncoder()
        self.y = self.y_encoder.fit_transform(y)

        self._splitter = Splitter(self.X,
                                  self.y,
                                  is_numerical,
                                  self.X_encoders,
                                  self.feature_names)
        self.tree_ = self._build(np.arange(n_samples),
                                 np.arange(self.n_features_idx))

        if prune:
            self.pruner.prune(self.tree_, X_test, y_test)

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
        X = check_array(X)
        return X[:, 0]**2


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features_idx]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features_idx]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
