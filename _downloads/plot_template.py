"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from id3 import Id3Estimator
from matplotlib import pyplot as plt

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = Id3Estimator()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
