from sklearn.datasets import load_breast_cancer
from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from id3 import Id3Estimator
from id3 import export_graphviz
from id3.data import load_simple
from id3.splitter import Splitter


y = np.array([0, 1, 2, 2, 3])
x_nominal_col = np.array([0, 0, 1, 0, 1])
x_numerical_col = np.array([1, 2, 5, 5, 1])
test_splitter = Splitter(None, None, None, None, None)


def test_entropy():
    x = 1 / 5. * np.log2(1 / (1 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.)) + \
        2 / 5. * np.log2(1 / (2 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.))
    assert_almost_equal(test_splitter._entropy(y), x)


def test_info_nominal():
    record = test_splitter._info_nominal(x_nominal_col, y)
    assert_equal(record.split_type, 1)
    assert_equal(record.attribute_counts.size, 4)
    assert_almost_equal(record.info, 1.3509775004326936)


def test_info_numerical():
    record = test_splitter._info_numerical(x_numerical_col, y)
    assert_equal(record.split_type, 0)
    assert_equal(record.attribute_counts.size, 10)
    assert_almost_equal(record.pivot, 2)
    assert_almost_equal(record.info, 0.9)


"""
def test_simple():
    X, y, targets = load_simple()
    id3Estimator.fit(X, y, targets)
    export_graphviz(id3Estimator.tree_, "cancer.dot")
"""


def test_fit():
    bunch = load_breast_cancer()

    id3Estimator = Id3Estimator()
    id3Estimator.fit(bunch.data, bunch.target, bunch.feature_names)
    assert_equal(id3Estimator.tree_.root.value, "worst perimeter")
    assert_equal(len(id3Estimator.tree_.classification_nodes), 64)
    assert_equal(len(id3Estimator.tree_.feature_nodes), 63)
    export_graphviz(id3Estimator.tree_, "cancer.dot")

    id3Estimator = Id3Estimator(max_depth=2)
    id3Estimator.fit(bunch.data, bunch.target, bunch.feature_names)
    assert_equal(id3Estimator.tree_.root.value, "worst perimeter")
    assert_equal(len(id3Estimator.tree_.classification_nodes), 4)
    assert_equal(len(id3Estimator.tree_.feature_nodes), 3)

    id3Estimator = Id3Estimator(min_samples_split=20)
    id3Estimator.fit(bunch.data, bunch.target, bunch.feature_names)
    assert_equal(id3Estimator.tree_.root.value, "worst perimeter")
    assert_equal(len(id3Estimator.tree_.classification_nodes), 35)
    assert_equal(len(id3Estimator.tree_.feature_nodes), 34)


def test_prune():
    non_pruning_estimator = Id3Estimator(prune=False)
    pruning_estimator = Id3Estimator(prune=True)
    bunch = load_breast_cancer()
    non_pruning_estimator.fit(bunch.data, bunch.target, bunch.feature_names)
    pruning_estimator.fit(bunch.data, bunch.target, bunch.feature_names)

    class_nodes = len(non_pruning_estimator.tree_.classification_nodes)
    feature_nodes = len(non_pruning_estimator.tree_.feature_nodes)

    pruned_class_nodes = len(pruning_estimator.tree_.classification_nodes)
    pruned_feature_nodes = len(pruning_estimator.tree_.feature_nodes)

    assert_equal(class_nodes > pruned_class_nodes, True)
    assert_equal(feature_nodes > pruned_feature_nodes, True)
