"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import numbers
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .tree import TreeBuilder, Tree
from .splitter import Splitter
from .utils import check_numerical_array, ExtendedLabelEncoder


class Id3Estimator(BaseEstimator):
    """ A template estimator for calculating ID3 decision trees.

    Parameters
    ----------
    max_depth : int, optional
        max depth of features
    min_samples_split : int, optional, default=2
        min samples to split on
    prune : bool, optional
        set to True to post-prune the tree
    gain_ratio : bool, optional
        use gain ratio on split calculations
    """
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 prune=False,
                 gain_ratio=False,
                 min_entropy_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.prune = prune
        self.gain_ratio = gain_ratio
        self.min_entropy_decrease = min_entropy_decrease

    def fit(self, X, y, check_input=True):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        check_input : bool
            check if the input for numerical features

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

        if isinstance(self.min_entropy_decrease,
                      (numbers.Integral, np.integer)):
            min_entropy_decrease = (0 if self.min_entropy_decrease < 0
                                    else self.min_entropy_decrease)
        else:
            min_entropy_decrease = 0

        n_samples, self.n_features = X_.shape
        self.is_numerical = [False] * self.n_features
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
                             self.gain_ratio)
        self.builder = TreeBuilder(splitter_,
                                   self.X_encoders,
                                   self.y_encoder,
                                   n_samples,
                                   self.n_features,
                                   self.is_numerical,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_entropy_decrease=min_entropy_decrease,
                                   prune=self.prune)
        self.tree_ = Tree()
        if self.prune:
            self.builder.build(self.tree_, self.X, self.y, X_test, y_test)
        else:
            self.builder.build(self.tree_, self.X, self.y)

        return self

    def predict(self, X):
        """ A predicting examples based on the previous fit.

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
        return self.builder._predict(self.tree_, X)
