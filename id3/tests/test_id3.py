import numpy as np
from numpy.testing import assert_almost_equal
from id3 import Id3Estimator
from id3.data import load_contact_lenses, load_will_wait
from sklearn.preprocessing import LabelEncoder
from id3 import export_graphviz

id3Estimator = Id3Estimator()


def test_entropy():
    y = np.array([0, 1, 2, 2, 3])
    x = 1 / 5. * np.log2(1 / (1 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.)) + \
        2 / 5. * np.log2(1 / (2 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.))
    assert_almost_equal(id3Estimator._entropy(y), x)


def test_split():
    '''
    X, y, targets = load_contact_lenses()
    X = np.apply_along_axis(LabelEncoder().fit_transform, axis=0, arr=X)
    y = LabelEncoder().fit_transform(y)
    feature_index = id3Estimator._split(X, y)[0]
    assert_almost_equal(feature_index, 3)
    X = X[X[:, feature_index] == 1]
    y = y[np.where(X[:, feature_index] == 1)]
    feature_index = id3Estimator._split(X, y)[0]
    assert_almost_equal(feature_index, 2)
    '''

def test_fit():
    X, y, targets = load_contact_lenses()
    estimator = Id3Estimator()
    estimator.fit(X, y, targets)
    export_graphviz(estimator.tree_)
