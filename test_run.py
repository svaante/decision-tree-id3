from id3 import Id3Estimator
from id3.data.load_data import load_data
from id3.export import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import time


def run():
    X, y, targets = load_data("simple4.arff")
    id3Estimator = Id3Estimator(prune=True, gain_ratio=True, is_repeating=True)
    t = time.time()
    id3Estimator.fit(X, y)

    """
    X_test, y_test, targets = load_data("car_test.arff")
    y_pred = id3Estimator.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    """

    print("Model done: {}".format(time.time() - t))
    export_graphviz(id3Estimator.tree_, "test.dot", feature_names=None)
    print(len(id3Estimator.tree_.classification_nodes))
    print(len(id3Estimator.tree_.classification_nodes) + len(id3Estimator.tree_.feature_nodes))


if __name__ == '__main__':
    run()
