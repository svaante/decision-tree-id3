from ..data.load_data import load_data
from .. import export_graphviz, export_text
from .. import Id3Estimator
from numpy.testing import assert_equal


test_tree = ('digraph ID3_Tree {\n'
             '"0" [shape=box, style=filled, label="0\n'
             'Info: 2.32\n'
             'Entropy: 3.32\n'
             'Dominant class: [0 1]\n'
             '", weight=1]\n'
             '"1" [shape=box, style=filled, label="1\n'
             'Info: 1.35\n'
             'Entropy: 2.32\n'
             'Dominant class: [0 1]\n'
             '", weight=2]\n'
             '0 -> 1 [ label = "<=9.00"];\n'
             '"2" [shape=box, style=filled, label="0\n'
             '(1/1) \n'
             '", weight=3]\n'
             '1 -> 2 [ label = "<=4.00"];\n'
             '"3" [shape=box, style=filled, label="2\n'
             '(1/2) \n'
             '", weight=3]\n'
             '1 -> 3 [ label = ">4.00"];\n'
             '"4" [shape=box, style=filled, label="1\n'
             'Info: 1.35\n'
             'Entropy: 2.32\n'
             'Dominant class: [5 1]\n'
             '", weight=2]\n'
             '0 -> 4 [ label = ">9.00"];\n'
             '"5" [shape=box, style=filled, label="5\n'
             '(1/1) \n'
             '", weight=3]\n'
             '4 -> 5 [ label = "<=14.00"];\n'
             '"6" [shape=box, style=filled, label="7\n'
             '(1/2) \n'
             '", weight=3]\n'
             '4 -> 6 [ label = ">14.00"];\n'
             '{rank=same; 0;};\n'
             '{rank=same; 1;4;};\n'
             '{rank=same; 2;3;5;6;};\n'
             '}')

test_tree_text = ('\n0 <=9.00\n'
                  '|   1 <=4.00: 0  (1/1)\n'
                  '|   1 >4.00: 2 (1/2)\n'
                  '0 >9.00\n'
                  '|   1 <=14.00: 5 (1/1)\n'
                  '|   1 >14.00: 7 (1/2)\n')


def test_export_graphviz():
    X, y, targets = load_data("test.csv")

    estimator = Id3Estimator()
    estimator.fit(X, y)

    tree = export_graphviz(estimator.tree_, extensive=True)
    actual = "".join(tree.to_string().split())
    desired = "".join(test_tree.split())
    assert_equal(actual, desired)


def test_export_text():
    X, y, targets = load_data("test.csv")

    estimator = Id3Estimator()
    estimator.fit(X, y)

    tree = export_text(estimator.tree_)
    actual = "".join(tree.split())
    desired = "".join(test_tree_text.split())
    assert_equal(actual, desired)
