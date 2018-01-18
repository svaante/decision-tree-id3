import numpy as np

from .node import Node
from .utils import unique
from .splitter import CalcRecord, SplitRecord


class Tree():
    """Class for storing the actual tree data"""

    def __init__(self,
                 root=None,
                 X_encoders=None,
                 y_encoder=None):
        self.root = root
        self.X_encoders = X_encoders
        self.y_encoder = y_encoder


class BaseBuilder():
    """Base class for different methods of building decision trees."""

    def build(self, tree, X, y):
        """Build a decision tree from data X and classifications y."""
        pass

    def _predict(self, tree, X, y=None):
        pass

    def _prune(self, tree, node):
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
                 min_entropy_decrease=0,
                 prune=False,
                 is_repeating=False):
        self.splitter = splitter
        self.y_encoder = y_encoder
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = y_encoder.classes_.size
        self.is_numerical = is_numerical
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_entropy_decrease = min_entropy_decrease
        self.prune = prune
        self.is_repeating = is_repeating

    def build(self, tree, X, y, X_test=None, y_test=None):
        self.X = X
        self.y = y
        tree.root = self._build(tree, np.arange(self.n_samples),
                                np.arange(self.n_features))
        if self.prune:
            if X_test is None or y_test is None:
                raise ValueError("Can't prune tree without validation data")
            self._predict(tree, X_test, y_test)
            self._prune(tree.root, tree)

    def _build(self, tree, examples_idx, features_idx, depth=0):
        items, counts = unique(self.y[examples_idx])
        if (features_idx.size == 0
                or items.size == 1
                or examples_idx.size < self.min_samples_split
                or depth >= self.max_depth):
            node = self._class_node(items, counts)
            return node

        calc_record = self.splitter.calc(examples_idx, features_idx)

        if (calc_record is None
                or calc_record.info < self.min_entropy_decrease):
            node = self._class_node(items, counts)
            return node

        split_records = self.splitter.split(examples_idx, calc_record)

        features_idx = np.compress(calc_record.alive_features, features_idx)
        if not self.is_repeating:
            features_idx = np.delete(features_idx,
                                     np.where(features_idx ==
                                              calc_record.feature_idx))
        root = Node(calc_record.feature_idx,
                    is_feature=True,
                    details=calc_record,
                    item_count=(items, counts))
        for record in split_records:
            if record.size == 0:
                node = self._class_node(items, counts)
                root.add_child(node, record)
            else:
                root.add_child(self._build(tree, record.bag,
                               features_idx, depth+1),
                               record)
        return root

    def _class_node(self, items, counts):
        classification = items[np.argmax(counts)]
        node = Node(classification, item_count=(items, counts))
        return node

    def _prune(self, node, tree):
        if node.is_feature:
            node.predicts = np.zeros(self.n_classes)
            n_children_correct = 0
            for child, _ in node.children:
                self._prune(child, tree)
                if child.predicts is not None:
                    node.predicts += child.predicts
                    n_children_correct += child.n_correct_predicts
                    child.predicts = None
                    child.n_correct_predicts = 0
            n_predicts = np.sum(node.predicts)
            if n_predicts > 0:
                max_class = np.argmax(node.predicts)
                children_error_rate = np.true_divide(n_predicts
                                                     - n_children_correct,
                                                     n_predicts)
                node_error_rate = np.true_divide(n_predicts
                                                 - node.predicts[max_class],
                                                 n_predicts)

                if node_error_rate < children_error_rate:
                    node.is_feature = False
                    node.value = max_class
                    node.n_correct_predicts = node.predicts[max_class]
                    node.children = []
                else:
                    node.n_correct_predicts = n_children_correct

    def _predict(self, tree, X, y=None):
        ret = np.empty(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            node = tree.root
            while(node.is_feature):
                value = x[node.details.feature_idx]
                for child, split_record in node.children:
                    if split_record.calc_record.split_type == CalcRecord.NOM:
                        if split_record.value_encoded == value:
                            node = child
                            break
                    elif (split_record.calc_record.split_type
                          == CalcRecord.NUM):
                        if (split_record.value_encoded ==
                            SplitRecord.GREATER and
                                value > split_record.calc_record.pivot):
                            node = child
                            break
                        elif (split_record.value_encoded ==
                              SplitRecord.LESS and
                              value <= split_record.calc_record.pivot):
                            node = child
                            break
            ret[i] = node.value
            if y is not None:
                node.add_predict_result(y[i], self.n_classes)
        return ret

    def _predict_proba(self, tree, X, y=None):
        ret = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        for i, x in enumerate(X):
            node = tree.root
            while(node.is_feature):
                value = x[node.details.feature_idx]
                for child, split_record in node.children:
                    if split_record.calc_record.split_type == CalcRecord.NOM:
                        if split_record.value_encoded == value:
                            node = child
                            break
                    elif (split_record.calc_record.split_type
                          == CalcRecord.NUM):
                        if (split_record.value_encoded ==
                            SplitRecord.GREATER and
                                value > split_record.calc_record.pivot):
                            node = child
                            break
                        elif (split_record.value_encoded ==
                              SplitRecord.LESS and
                              value <= split_record.calc_record.pivot):
                            node = child
                            break
            items, counts = node.item_count
            if counts.size > 0:
                for item, count in zip(items, counts):
                    ret[i, item] = count / np.sum(counts)
        return ret
