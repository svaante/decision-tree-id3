import numpy as np
from sklearn.metrics import accuracy_score

from .node import Node
from .utils import unique
from .splitter import CalcRecord, SplitRecord


class Tree():
    """Class for storing the actual tree data"""

    def __init__(self,
                 root=None,
                 classification_nodes=None,
                 feature_nodes=None):
        self.root = root
        if classification_nodes is None:
            classification_nodes = []
        if feature_nodes is None:
            feature_nodes = []
        self.classification_nodes = classification_nodes
        self.feature_nodes = feature_nodes


class BaseBuilder():
    """Base class for different methods of building decision trees."""

    def build(self, tree, X, y):
        """Build a decision tree from data X and classifications y."""
        pass

    def prune(self, tree):
        pass


class TreeBuilder(BaseBuilder):
    """Build a decision tree using the default strategy"""

    def __init__(self,
                 splitter,
                 y_encoder,
                 n_samples,
                 n_features,
                 is_numerical,
                 max_depth=None,
                 min_samples_split=1,
                 prune=False):
        self.splitter = splitter
        self.y_encoder = y_encoder
        self.n_samples = n_samples
        self.n_features = n_features
        self.is_numerical = is_numerical
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.prune = prune

    def build(self, tree, X, y, X_test=None, y_test=None):
        self.X = X
        self.y = y
        tree.root = self._build(tree, np.arange(self.n_samples),
                                np.arange(self.n_features))
        if self.prune:
            self._prune(tree, X_test, y_test)

    def _build(self, tree, examples_idx, features_idx, depth=0):
        items, counts = unique(self.y[examples_idx])
        classification = items[np.argmax(counts)]
        classification_name = self.y_encoder.inverse_transform(classification)
        if features_idx.size == 0 or items.size == 1:
            node = Node(classification_name)
            tree.classification_nodes.append(node)
            return node

        calc_record = self.splitter.calc(examples_idx, features_idx)
        split_records = self.splitter.split(examples_idx, calc_record)
        new_features_idx = np.delete(features_idx,
                                     np.where(features_idx ==
                                              calc_record.feature_idx))
        root = Node(calc_record.feature_name,
                    is_feature=True,
                    details=calc_record)
        tree.feature_nodes.append(root)
        for record in split_records:
            if record.size == 0:
                node = Node(classification_name)
                tree.classification_nodes.append(node)
                root.add_child(node, record)
            else:
                root.add_child(self._build(tree, record.bag,
                               new_features_idx, depth+1),
                               record)
        return root

    def _prune(self, tree, X_test, y_test):
        y_pred = self._predict(tree, X_test)
        base_score = accuracy_score(y_pred, y_test)
        for node in tree.feature_nodes:
            if not node.is_feature:
                continue
            encoded_class = node.details.class_counts[0, 0]
            decoded_class = self.y_encoder.inverse_transform(encoded_class)
            tmp_value = node.value
            node.value = decoded_class
            node.is_feature = False
            y_pred = self._predict(tree, X_test)
            new_score = accuracy_score(y_pred, y_test)
            if new_score < base_score:
                node.value = tmp_value
                node.is_feature = True
            else:
                node.children = []

    def _predict(self, tree, X):
        X_ = np.zeros(X.shape)
        ret = np.empty(X.shape[0], dtype=X.dtype)
        for i in range(self.n_features):
            if self.is_numerical[i]:
                X_[:, i] = X[:, i]
            else:
                X_[:, i] = self.X_encoders[i].transform(X[:, i])
        for i, x in enumerate(X_):
            node = tree.root
            while(node.is_feature):
                value = x[node.details.feature_idx]
                for child, split_record in node.children:
                    if split_record.calc_record.split_type == CalcRecord.NOM:
                        if split_record.value_encoded == value:
                            node = child
                            break
                    elif split_record.calc_record.split_type == CalcRecord.NUM:
                        if (split_record.value_encoded ==
                            SplitRecord.GREATER and
                                value >= split_record.calc_record.pivot):
                            node = child
                            break
                        elif (split_record.value_encoded ==
                              SplitRecord.LESS and
                              value < split_record.calc_record.pivot):
                            node = child
                            break
            ret[i] = node.value
        return ret
        pass
