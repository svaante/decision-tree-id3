"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import LabelEncoder

'''
def gain_after_tmp(feature_values, y):
    """ Gain for feature feature_values p(a)H(a)

    Parameters
    ----------
    feature_values : nparray attribute column
    y : nparray class array
    """
    def single_gain(p, value):
        return p * entropy(y[feature_values == value])

    n = feature_values.shape
    unique, count = np.unique(feature_values, return_counts=True)
    gain_ = np.vectorize(single_gain)
    gain = np.sum(gain_(count, unique))
    return gain * np.true_divide(1, n)
'''


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
        """ Entropy for the the classes in the array y
        \sum_{x \in X} p(x) \log_{2}(1/p(x))

        Parameters
        ----------
        y : nparray of shape [n_remaining attributes] containing the class
            names
        """
        n = y.shape[0]
        if n <= 0:
            return 0
        _, count = np.unique(y, return_counts=True)
        p = np.true_divide(count, n)
        return np.sum(np.multiply(p, np.log2(np.reciprocal(p))))

    def _gain_after(self, feature_values, y):
        """ Gain for feature feature_values p(a)H(a)

        Parameters
        ----------
        feature_values : nparray attribute column
        y : nparray class array
        """
        gain = 0
        n = feature_values.shape
        unique, count = np.unique(feature_values, return_counts=True)
        for value, p in zip(unique, count):
            gain += p * self._entropy(y[feature_values == value])
        return gain * np.true_divide(1, n)

    def _split(self, X, y):
        """ Returns feture index for max gain split

        Parameters
        ----------
        X : nparray feature 2d array
        y : nparray class array
        """
        return np.argmin(np.apply_along_axis(self._gain_after, 0, X, y))

    def fit(self, X, y):
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
        n_features : int
            The number of features when ``fit`` is performed.

        X_encoders : list
            List of LabelEncoders that transforms input from labels to binary encodings and vice versa.

        y_encoder : LabelEncoder
            LabelEncoders that transforms output from labels to binary encodings and vice versa.

        Returns
        -------
        self : object
            Returns self.
        """
        X_, y = check_X_y(X, y)
        n_samples, self.n_features = X.shape
        X = np.zeros(X.shape, dtype=np.int)

        self.X_encoders = [LabelEncoder() for _ in range(self.n_features)]
        for i in range(self.n_features):
            X[:, i] = self.X_encoders[i].fit_transform(X_[:, i])
        self.y_encoder = LabelEncoder()
        y = self.y_encoder.fit_transform(y)

        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
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
    X_ : array, shape = [n_samples, n_features]
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
        X : array-like, shape = [n_samples, n_features]
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
