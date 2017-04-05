import numpy as np
from numpy.testing import assert_almost_equal
from id3 import Id3Estimator
from id3.id3 import entropy, information_gain
from id3.data import load_contact_lenses
from sklearn.preprocessing import LabelEncoder


def test_entropy():
    y = np.array([0, 1, 2, 2, 3])
    x = 1 / 5. * np.log2(1 / (1 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.)) + \
        2 / 5. * np.log2(1 / (2 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.))
    assert_almost_equal(entropy(y), x)


def test_information_gain():
    X, y, targets = load_contact_lenses()
    print(X)
    X = np.apply_along_axis(LabelEncoder().fit_transform, axis=0, arr=X)
    y = LabelEncoder().fit_transform(y)
    a = [1, 2, 3]
    print(X)
    print(X[:, a])
    assert_almost_equal(information_gain(X, y, a), 1)


def test_fit():
    X, y, targets = load_contact_lenses()
    estimator = Id3Estimator()
    estimator.fit(X, y)
