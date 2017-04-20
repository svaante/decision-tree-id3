from sklearn.utils.estimator_checks import check_estimator
from id3 import Id3Estimator


def test_estimator():
    return check_estimator(Id3Estimator)
