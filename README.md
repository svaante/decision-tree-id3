#project-template - A template for scikit-learn extensions

[![Travis Status](https://travis-ci.org/svaante/decision-tree-id3.svg?branch=master)](https://travis-ci.org/svaante/decision-tree-id3)
[![Coveralls Status](https://coveralls.io/repos/svaante/decision-tree-id3/badge.svg?branch=master&service=github)](https://coveralls.io/r/svaante/decision-tree-id3)
[![CircleCI Status](https://circleci.com/gh/svaante/decision-tree-id3.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/svaante/decision-tree-id3/tree/master)

**decision-tree-id3**
decision-tree-id3 is a module created to derive decision trees using the ID3 algorithm. It is written to be compatible with Scikit-learn's API using the guidelines for Scikit-learn-contrib. It is licensed under the 3-clause BSD license.

[scikit-learn](http://scikit-learn.org/) 
compatible extensions.


It aids development of estimators that can be used in scikit-learn pipelines
and (hyper)parameter search, while facilitating testing (including some API
compliance), documentation, open source development, packaging, and continuous
integration.

## Important Links
HTML Documentation - https://svaante.github.io/decision-tree-id3

## Installation
### Dependencies
  - Python (>= 2.7 or >= 3.3)
  - NumPy (>= 1.6.1)
  - Scikit-learn (>= 0.17)
  
The package by itself comes with a single estimator Id3Estimator.
To install the module:
```
pip install decision-tree-id3
```
or clone the project using:
```
git clone https://github.com/svaante/decision-tree-id3.git
cd decision-tree-id3
python setup.py install
```

## Usage
If the installation is successful, you should be able to execute the following in Python:
```python
>>> from sklearn.datasets import load_breast_cancer
>>> from id3 import Id3Estimator
>>> from id3 import export_graphviz

>>> bunch = load_breast_cancer()
>>> estimator = Id3Estimator()
>>> estimator.fit(bunch.data, bunch.target)
>>> export_graphviz(estimator.tree_, 'tree.dot', bunch.feature_names)
```
And to generate a PDF of the decision tree using GraphViz:
```shell
dot -Tpdf tree.dot -o tree.pdf
```

There are a number of different default parameters to control the growth of the tree:
  - max_depth, the max depth of the tree.
  - min_samples_split, the minimum number of samples in a split to be considered.
  - prune, if the tree should be post-pruned to avoid overfitting and cut down on size.
  - gain_ratio, if the algorithm should use gain ratio when splitting the data.
  - min_entropy_decrease, the minimum decrease in entropy to consider a split.
  - is_repeating, repeat the use of features.
  
  For more in depth information see the documentation https://svaante.github.io/decision-tree-id3
