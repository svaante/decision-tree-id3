from sklearn.datasets import load_breast_cancer
from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from id3 import Id3Estimator
from id3.splitter import Splitter, CalcRecord


y = np.array([0, 1, 2, 2, 3])
x_nominal_col = np.array([0, 0, 1, 0, 1])
x_numerical_col = np.array([1, 2, 5, 5, 1])

X = np.array([[45, "male", "private", "m"],
              [50, "female", "private", "m"],
              [61, "other", "public", "b"],
              [40, "male", "private", "none"],
              [34, "female", "private", "none"],
              [33, "male", "public", "none"],
              [43, "other", "private", "m"],
              [35, "male", "private", "m"],
              [34, "female", "private", "m"],
              [35, "male", "public", "m"],
              [34, "other", "public", "m"],
              [34, "other", "public", "b"],
              [34, "female", "public", "b"],
              [34, "male", "public", "b"],
              [34, "female", "private", "b"],
              [34, "male", "private", "b"],
              [34, "other", "private", "b"]])

X_nom_test = np.array([[25, "male", "private", "m"],
                       [30, "female", "private", "b"],
                       [71, "other", "public", "none"],
                       [19, "female", "private", "m"]])

y_nom_test = np.array(['(23k,30k)',
                       '(23k,30k)',
                       '(13k,15k)',
                       '(23k,30k)'])

y_nom = np.array(["(30k,38k)",
                  "(30k,38k)",
                  "(30k,38k)",
                  "(13k,15k)",
                  "(13k,15k)",
                  "(13k,15k)",
                  "(23k,30k)",
                  "(23k,30k)",
                  "(23k,30k)",
                  "(15k,23k)",
                  "(15k,23k)",
                  "(15k,23k)",
                  "(15k,23k)",
                  "(15k,23k)",
                  "(23k,30k)",
                  "(23k,30k)",
                  "(23k,30k)"])

test_splitter = Splitter(None, None, None, None)

bc_sample = np.array([20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869,
                   0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08,
                   0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
                   24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416,
                   0.186, 0.275, 0.08902]).reshape(1, -1)

def test_intrinsic_value():
    c1 = CalcRecord(None, None, attribute_counts=np.array([5, 4, 5]))
    c2 = CalcRecord(None, None, attribute_counts=np.array([7, 7]))
    c3 = CalcRecord(None, None, attribute_counts=np.array([8, 6]))
    c4 = CalcRecord(None, None, attribute_counts=np.array([4, 6, 4]))
    assert_almost_equal(test_splitter._intrinsic_value(c1), 1.577, 3)
    assert_almost_equal(test_splitter._intrinsic_value(c2), 1.000, 3)
    assert_almost_equal(test_splitter._intrinsic_value(c3), 0.985, 3)
    assert_almost_equal(test_splitter._intrinsic_value(c4), 1.557, 3)


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
    assert_almost_equal(record.pivot, 1.5, 2)
    assert_almost_equal(record.info, 0.95, 2)
    assert_almost_equal(record.attribute_counts, [2, 3])


def test_numerical_split():
    bunch = load_breast_cancer()

    id3Estimator = Id3Estimator()
    id3Estimator.fit(bunch.data, bunch.target)
    splitter = id3Estimator.builder_.splitter
    record = splitter.calc(np.array(list(range(bunch.target.shape[0]))),
                           np.array(list(range(bunch.data.shape[1]))))
    less = np.sum(bunch.data[:, record.feature_idx] <= record.pivot)
    more = bunch.data[:, record.feature_idx].shape[0] - less
    split = splitter.split(np.array(list(range(bunch.target.shape[0]))),
                           record)
    assert_almost_equal(len(split[0].bag), less)
    assert_almost_equal(len(split[1].bag), more)


def test_fit():
    bunch = load_breast_cancer()

    id3Estimator = Id3Estimator()
    id3Estimator.fit(bunch.data, bunch.target)
    assert_equal(id3Estimator.tree_.root.value, 22)

    id3Estimator = Id3Estimator(max_depth=2)
    id3Estimator.fit(bunch.data, bunch.target)
    assert_equal(id3Estimator.tree_.root.value, 22)

    id3Estimator = Id3Estimator(min_samples_split=20)
    id3Estimator.fit(bunch.data, bunch.target)
    assert_equal(id3Estimator.tree_.root.value, 22)


def test_nominal():
    id3Estimator = Id3Estimator()
    id3Estimator.fit(X, y_nom)

    assert_equal(id3Estimator.tree_.root.value, 3)
    predict = id3Estimator.predict(X_nom_test)
    assert_equal(predict, y_nom_test)


def test_gain_ratio():
    id3Estimator = Id3Estimator(gain_ratio=True)
    bunch = load_breast_cancer()
    id3Estimator.fit(bunch.data, bunch.target)

    assert_equal(id3Estimator.tree_.root.value, 23)


def test_prune():
    id3estimator = Id3Estimator(prune=True)
    bunch = load_breast_cancer()
    id3estimator.fit(bunch.data, bunch.target)

def test_predict():
    estimator = Id3Estimator()
    bunch = load_breast_cancer()
    estimator.fit(bunch.data, bunch.target)
    assert_almost_equal(estimator.predict(bunch.data), bunch.target)
    assert_almost_equal(estimator.predict(bc_sample), 0)

def test_predict_proba():
    estimator = Id3Estimator()
    bunch = load_breast_cancer()
    estimator.fit(bunch.data, bunch.target)
    # Test shape of probability data structure using breast cancer data
    probs = estimator.predict_proba(bunch.data)
    assert_equal(probs.shape[0],bunch.data.shape[0])
    assert_equal(probs.shape[1],estimator.tree_.y_encoder.classes_.shape[0])
    # Test probability values using sample data (as per test_predict method)
    probs = estimator.predict_proba(bc_sample)
    assert probs[0,0] >= 0.5
    assert probs[0,1] < 0.5
    assert_equal(np.sum(probs[0]),1.0)
