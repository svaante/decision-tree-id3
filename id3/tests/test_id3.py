from id3 import Id3Estimator
from sklearn.datasets import load_breast_cancer
from id3 import export_graphviz

id3Estimator = Id3Estimator(pruner="ReducedError")


"""
def test_entropy():
    y = np.array([0, 1, 2, 2, 3])
    x = 1 / 5. * np.log2(1 / (1 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.)) + \
        2 / 5. * np.log2(1 / (2 / 5.)) + 1 / 5. * np.log2(1 / (1 / 5.))
    assert_almost_equal(Id3Estimator()._entropy(y), x)
"""


def test_breast_cancer():
    bunch = load_breast_cancer()

    id3Estimator.fit(bunch.data, bunch.target, bunch.feature_names)
    export_graphviz(id3Estimator.tree_, "cancer.dot")
