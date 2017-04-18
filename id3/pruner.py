from sklearn.externals import six
from abc import ABCMeta, abstractmethod

class BasePruner(six.with_metaclass(ABCMeta)):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prune(self, tree):
        pass

class ErrorPruner(BasePruner):

    def __init__(self):
        pass

    def prune(self, tree):
        print("pruning")

class CostPruner(BasePruner):

    def prune(self, tree):
        print("pruning")
