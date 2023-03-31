"""
==========================
Graph export from Estimator
==========================

An example graph export of :class:`id3.id3.Id3Estimator` with
:file:`id3.export.export_graphviz`


$ dot -T png out.dot -o out.png

.. figure::  /_static/out.png
   :align:   center

"""

from src.id3 import Id3Estimator, export_graphviz
import numpy as np

feature_names = ["age",
                 "gender",
                 "sector",
                 "degree"]

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

y = np.array(["(30k,38k)",
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

clf = Id3Estimator()
clf.fit(X, y, check_input=True)

export_graphviz(clf.tree_, "out.dot", feature_names)
