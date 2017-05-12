import numpy as np
import numbers
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .tree import TreeBuilder, Tree
from .splitter import Splitter
from .utils import check_numerical_array, ExtendedLabelEncoder


class Id3Estimator(BaseEstimator):
    """A decision tree estimator for deriving ID3 decision trees.

    Parameters
    ----------
    max_depth : int, optional
        max depth of features.
    min_samples_split : int, optional (default=2)
        min samples to split on.
    prune : bool, optional (default=False)
        set to True to prune the tree.
    gain_ratio : bool, optional (default=False)
        use gain ratio on split calculations.
    is_repeating: bool, optional (default=False)
        use repeating features.

    Attributes
    ----------
    max_depth : int
    min_samples_split : int
    prune : bool
    gain_ratio : bool
    min_entropy_decrease : float
    is_repeating : bool
    """
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 prune=False,
                 gain_ratio=False,
                 min_entropy_decrease=0.0,
                 is_repeating=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.prune = prune
        self.gain_ratio = gain_ratio
        self.min_entropy_decrease = min_entropy_decrease
        self.is_repeating = is_repeating

    def fit(self, X, y, check_input=True):
        """Build a decision tree based on samples X and
        corresponding classifications y.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        check_input : bool (default=True)
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

        is_numerical : bool array of size [n_features]
            Array flagging which features that are asumed to be numerical

        builder_ : TreeBuilder
            Instance of the tree builder

        tree_ : Tree
            Instance of the build tree

        Returns
        -------
        self : object
            Returns self.
        """
        X_, y_ = check_X_y(X, y)
        self.y_encoder = ExtendedLabelEncoder()
        y_ = self.y_encoder.fit_transform(y_)

        max_np_int = np.iinfo(np.int32).max
        if not isinstance(self.max_depth, (numbers.Integral, np.integer)):
            max_depth = max_np_int
        else:
            max_depth = self.max_depth

        if isinstance(self.min_samples_split,
                      (numbers.Integral, np.integer)):
            min_samples_split = (1 if self.min_samples_split < 1
                                 else self.min_samples_split)
        else:
            min_samples_split = 1

        if isinstance(self.min_entropy_decrease,
                      (np.float, np.integer)):
            min_entropy_decrease = (0 if self.min_entropy_decrease < 0
                                    else self.min_entropy_decrease)
        else:
            min_entropy_decrease = 0

        _, self.n_features = X_.shape
        self.is_numerical = [False] * self.n_features
        X_tmp = np.zeros(X_.shape, dtype=np.float32)
        self.X_encoders = [ExtendedLabelEncoder() for _ in
                           range(self.n_features)]
        for i in range(self.n_features):
            if check_input and check_numerical_array(X_[:, i]):
                self.is_numerical[i] = True
                X_tmp[:, i] = X_[:, i]
            else:
                X_tmp[:, i] = self.X_encoders[i].fit_transform(X_[:, i])
        X_ = X_tmp
        if self.prune:
            X_, X_test, y_, y_test = train_test_split(X_,
                                                      y_,
                                                      test_size=0.3)

        splitter = Splitter(X_,
                            y_,
                            self.is_numerical,
                            self.X_encoders,
                            self.gain_ratio)

        self.builder_ = TreeBuilder(splitter,
                                    self.y_encoder,
                                    X_.shape[0],
                                    self.n_features,
                                    self.is_numerical,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_entropy_decrease=min_entropy_decrease,
                                    prune=self.prune,
                                    is_repeating=self.is_repeating)
        self.tree_ = Tree(X_encoders=self.X_encoders,
                          y_encoder=self.y_encoder)
        if self.prune:
            self.builder_.build(self.tree_, X_, y_, X_test, y_test)
        else:
            self.builder_.build(self.tree_, X_, y_)

        return self

    def predict(self, X):
        """Predict class for every sample in X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features_idx]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
        """
        check_is_fitted(self, 'tree_')
        X = check_array(X)
        n_features = X.shape[1]
        if n_features != self.n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {}."
                             .format(self.n_features, n_features))

        X_ = np.empty(X.shape)
        for i in range(self.n_features):
            if self.is_numerical[i]:
                X_[:, i] = X[:, i]
            else:
                try:
                    X_[:, i] = self.X_encoders[i].transform(X[:, i])
                except ValueError as e:
                    raise ValueError('New attribute value not found in '
                                     'train data.')
        y = self.builder_._predict(self.tree_, X_)
        return self.y_encoder.inverse_transform(y)
