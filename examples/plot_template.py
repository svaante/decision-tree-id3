"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
from id3 import Id3Estimator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

bunch = load_breast_cancer()
clf = Id3Estimator()
X_train, X_test, y_train, y_test = train_test_split(bunch.data,
                                                    bunch.target,
                                                    test_size=0.2)
clf.fit(X_train, y_train)

plt.figure()
plt.scatter(X_train, y_train, label="data")
plt.plot(X_test, clf.predict(X_test), label="predict")
plt.xlabel("data")
plt.ylabel("target")
plt.show()
