import numpy as np
from numpy.testing import assert_equal, assert_raises
from id3.utils import check_numerical_array
from id3.utils import unique
from id3.utils import ExtendedLabelEncoder

numeric = np.arange(5)
nominal = np.array(["nom", "nom", "nom"])
error = np.arange(6).reshape((2, 3))
labels = np.array(["first", "second", "third"])


def test_check_numerical_array():
    assert_equal(check_numerical_array(numeric), True)
    assert_equal(check_numerical_array(nominal), False)
    assert_raises(ArithmeticError, check_numerical_array, error)


def test_unique():
    a = np.array([1, 1, 2, 5, 5, 1, 9, 10, 2, 5])
    un = np.array([1, 2, 5, 9, 10])
    count = np.array([3, 2, 3, 1, 1])
    assert_equal(unique(a), (un, count))


def test_extended_encoder():
    encoder = ExtendedLabelEncoder()
    encoder.fit(labels)
    assert_equal(labels, encoder.classes_)
    assert_equal(np.array([0, 1, 2]), encoder.encoded_classes_)
