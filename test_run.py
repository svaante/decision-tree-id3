from id3 import Id3Estimator
from id3.data.load_data import load_data
from id3.export import export_graphviz
import resource
import time
rsrc = resource.RLIMIT_DATA
soft, hard = resource.getrlimit(rsrc)


def run():
    X, y, targets = load_data('glass.arff')
    id3Estimator = Id3Estimator()
    t = time.time()
    id3Estimator.fit(X, y)
    print("Model done: {}".format(time.time() - t))
    print(len(id3Estimator.tree_.classification_nodes))
    print(len(id3Estimator.tree_.classification_nodes) + len(id3Estimator.tree_.feature_nodes))
    export_graphviz(id3Estimator.tree_, "test.dot")


if __name__ == '__main__':
    run()
