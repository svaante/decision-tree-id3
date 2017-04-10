"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import LabelEncoder

from .node import Node
from .utils import check_numerical_array

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

    def _info(self, feature_values, y):
        """ info for feature feature_values
        :math: p(a)H(a) :math: from
        https://en.wikipedia.org/wiki/ID3_algorithm

        Parameters
        ----------
        feature_values : np.array of shape [n remaining examples]
            containing feature values
        y : np.array of shape [n remaining examples]
            containing relevant class

        Returns
        -------
        : float
            information for remaining examples given feature
        """
        info = 0
        n = feature_values.shape[0]
        unique, count = np.unique(feature_values, return_counts=True)
        for value, p in zip(unique, count):
            info += p * self._entropy(y[feature_values == value])
        return info * np.true_divide(1, n)

    def _split(self, examples_idx, features_idx):
        """ Returns feature index for max info split

        Parameters
        ----------
        examples_idx: np.array
            data row(s) to be considered
        features_idx: np.array
            feature colum(s) to be considered

        Returns
        -------
        : int
            feature to split on in global self.x index
        : float
            min info value
        """
        X_ = self.X[np.ix_(examples_idx, features_idx)]
        y_ = self.y[examples_idx]
        info = np.apply_along_axis(self._info, 0, X_, y_)
        argmin_info = np.argmin(info)
        return features_idx[argmin_info], info[argmin_info]

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
        if features_idx.size == 0:
            return Node(unique[np.argmax(counts)], self.y_encoder)
        if unique.size == 1:
            return Node(unique[0], self.y_encoder)
        argmin, info = self._split(examples_idx, features_idx)
        encoder = self.X_encoders[argmin]
        values = encoder.transform(encoder.classes_)
        root = Node(argmin,
                    encoder,
                    self.feature_names[argmin] if self.feature_names is not None else None,
                    is_feature=True,
                    details={
                                'Entropy':
                                self._entropy(self.y[examples_idx]),
                                'Info':
                                info
                            })
        new_features_idx = np.delete(features_idx,
                                     np.where(features_idx == argmin))
        for value in values:
            new_X = self.X[np.ix_(examples_idx)]
            new_examples_idx = examples_idx[new_X[:, argmin] == value]
            if new_examples_idx.size != 0:
                root.add_child(self._build(new_examples_idx, new_features_idx),
                               value)
            else:
                root.add_child(Node(unique[np.argmax(counts)],
                                    self.y_encoder),
                               value)
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
        self.feature_names = feature_names
        numerical_values = []
        X_, y = check_X_y(X, y)
        n_samples, self.n_features_idx = X.shape
        if (self.feature_names is not None and
                len(self.feature_names) != (self.n_features_idx + 1)):
            raise ValueError(("feature_names needs to have the same "
                               "number of elements as features in X"),)
        self.X = np.zeros(X.shape, dtype=np.int)
        self.X_encoders = [LabelEncoder() for _ in range(self.n_features_idx)]
        for i in range(self.n_features_idx):
            if check_input and check_numerical_array(X_[:, i]):
                numerical_values.append(i)
            self.X[:, i] = self.X_encoders[i].fit_transform(X_[:, i])
        self.y_encoder = LabelEncoder()
        self.y = self.y_encoder.fit_transform(y)

        self.tree_ = self._build(np.arange(n_samples), np.arange(self.n_features_idx))
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
