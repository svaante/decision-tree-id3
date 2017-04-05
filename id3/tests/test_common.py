from sklearn.utils.estimator_checks import check_estimator
from id3 import (Id3Estimator, TemplateClassifier, TemplateTransformer)


def test_estimator():
    return check_estimator(Id3Estimator)


def test_classifier():
    return check_estimator(TemplateClassifier)


def test_transformer():
    return check_estimator(TemplateTransformer)
