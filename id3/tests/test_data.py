from numpy.testing import assert_almost_equal, assert_raises
import numpy as np
from id3.data import load_data
import uuid

X = np.arange(20).reshape(10, 2)
y = np.arange(10).reshape(10, )


def test_load_data():
    assert_raises(IOError, load_data.load_data, str(uuid.uuid4()))
    X_, y_, _ = load_data.load_data("test.csv", nominal=False)
    assert_almost_equal(X, X_)
    assert_almost_equal(y, y_)
