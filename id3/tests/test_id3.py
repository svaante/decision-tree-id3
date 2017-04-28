from sklearn.datasets import load_breast_cancer
from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from id3 import Id3Estimator
from id3 import export_graphviz
from id3.splitter import Splitter


y = np.array([0, 1, 2, 2, 3])
x_nominal_col = np.array([0, 0, 1, 0, 1])
x_numerical_col = np.array([1, 2, 5, 5, 1])
test_splitter = Splitter(None, None, None, None)


def test_entropy():
    x = 1 / 5. * np.log2(1 / (1 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.)) + \
        2 / 5. * np.log2(1 / (2 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.))
    assert_almost_equal(test_splitter._entropy(y), x)


def test_info_nominal():
    record = test_splitter._info_nominal(x_nominal_col, y)
    assert_equal(record.split_type, 1)
    assert_equal(record.attribute_counts.size, 2)
    assert_almost_equal(record.info, 1.3509775004326936)


def test_info_numerical():
    record = test_splitter._info_numerical(x_numerical_col, y)
    assert_equal(record.split_type, 0)
    assert_equal(record.attribute_counts.size, 2)
    assert_almost_equal(record.pivot, 2)
    assert_almost_equal(record.info, 0.95, 2)


def test_fit():
    bunch = load_breast_cancer()

    id3Estimator = Id3Estimator()
    id3Estimator.fit(bunch.data, bunch.target)
    assert_equal(id3Estimator.tree_.root.value, 22)
    assert_equal(len(id3Estimator.tree_.classification_nodes), 23)
    assert_equal(len(id3Estimator.tree_.feature_nodes), 22)
    export_graphviz(id3Estimator.tree_,
                    "cancer.dot",
                    feature_names=bunch.feature_names)

    id3Estimator = Id3Estimator(max_depth=2)
    id3Estimator.fit(bunch.data, bunch.target)
    assert_equal(id3Estimator.tree_.root.value, 22)
    assert_equal(len(id3Estimator.tree_.classification_nodes), 4)
    assert_equal(len(id3Estimator.tree_.feature_nodes), 3)

    id3Estimator = Id3Estimator(min_samples_split=20)
    id3Estimator.fit(bunch.data, bunch.target)
    assert_equal(id3Estimator.tree_.root.value, 22)
    assert_equal(len(id3Estimator.tree_.classification_nodes), 14)
    assert_equal(len(id3Estimator.tree_.feature_nodes), 13)

    id3Estimator = Id3Estimator(gain_ratio=True)
    id3Estimator.fit(bunch.data, bunch.target)
    export_graphviz(id3Estimator.tree_,
                    "cancer.dot",
                    feature_names=bunch.feature_names)


def test_gain_ratio():
    id3Estimator = Id3Estimator(gain_ratio=True)
    bunch = load_breast_cancer()
    id3Estimator.fit(bunch.data, bunch.target)
    export_graphviz(id3Estimator.tree_,
                    "cancer_gain_ratio.dot",
                    feature_names=bunch.feature_names)


def test_prune():
    estimator = Id3Estimator(prune=True)
    bunch = load_breast_cancer()
    estimator.fit(bunch.data, bunch.target)
    assert_equal(estimator.tree_.root is not None, True)
    assert_equal(len(estimator.tree_.classification_nodes) > 0, True)
    assert_equal(len(estimator.tree_.feature_nodes) > 0, True)
