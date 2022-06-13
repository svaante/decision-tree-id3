|Travis Status| |Coveralls Status| |CircleCI Status|

decision-tree-id3-fork
======================

This is a fork of decision-tree-id3. It is a module created to derive decision trees using the ID3 algorithm. It is written to be compatible with Scikit-learn's API using the guidelines for Scikit-learn-contrib. It is licensed under the 3-clause BSD license.

Important Links
---------------

Original HTML Documentation can be found on https://svaante.github.io/decision-tree-id3

Installation
------------

Dependencies
~~~~~~~~~~~~

-  Python (>= 2.7 or >= 3.3)
-  NumPy (>= 1.14.6)
-  Scikit-learn (>= 1.0)
-  Six (>= 0.9.0)

The package by itself comes with a single estimator Id3Estimator. To install the module:

::

    pip install decision-tree-id3-fork

or clone the project using:

::

    git clone https://github.com/DekwoKybon/decision-tree-id3.git
    cd decision-tree-id3
    python setup.py install

Usage
-----

If the installation is successful, you should be able to execute the
following in Python:

.. code:: python

    >>> from sklearn.datasets import load_breast_cancer
        >>> from src.id3 import Id3Estimator
        >>> from src.id3 import export_graphviz

        >>> bunch = load_breast_cancer()
        >>> estimator = Id3Estimator()
        >>> estimator = estimator.fit(bunch.data, bunch.target)
        >>> tree = export_graphviz(estimator.tree_, 'tree.dot', bunch.feature_names)


    And to generate a PDF of the decision tree using GraphViz:
    >>> from id3 import Id3Estimator
    >>> from id3 import export_graphviz

    >>> bunch = load_breast_cancer()
    >>> estimator = Id3Estimator()
    >>> estimator = estimator.fit(bunch.data, bunch.target)
    >>> tree = export_graphviz(estimator.tree_, 'tree.dot', bunch.feature_names)


And to generate a PDF of the decision tree using GraphViz:

.. code:: shell

    dot -Tpdf tree.dot -o tree.pdf

There are a number of different default parameters to control the growth
of the tree: - max\_depth, the max depth of the tree. -
min\_samples\_split, the minimum number of samples in a split to be
considered. - prune, if the tree should be post-pruned to avoid
overfitting and cut down on size. - gain\_ratio, if the algorithm should
use gain ratio when splitting the data. - min\_entropy\_decrease, the
minimum decrease in entropy to consider a split. - is\_repeating, repeat
the use of features.

For more in depth information see the documentation
https://svaante.github.io/decision-tree-id3

.. |Travis Status| image:: https://travis-ci.org/svaante/decision-tree-id3.svg?branch=master
   :target: https://travis-ci.org/svaante/decision-tree-id3
.. |Coveralls Status| image:: https://coveralls.io/repos/svaante/decision-tree-id3/badge.svg?branch=master
   :target: https://coveralls.io/r/svaante/decision-tree-id3
.. |CircleCI Status| image:: https://circleci.com/gh/svaante/decision-tree-id3.svg?style=shield&circle-token=:circle-token
   :target: https://circleci.com/gh/svaante/decision-tree-id3/tree/master
