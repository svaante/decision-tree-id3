import numpy as np
from numpy.testing import assert_equal, assert_raises
from id3.utils import check_numerical_array

numeric = np.arange(5)
nominal = np.array(['nom', 'nom', 'nom'])
error = np.arange(6).reshape((2, 3))


def test_check_numerical_array():
    assert_equal(check_numerical_array(numeric), True)
    assert_equal(check_numerical_array(nominal), False)
    assert_raises(ArithmeticError, check_numerical_array, error)
