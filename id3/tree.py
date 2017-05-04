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
                 X_encoders,
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
        self.X_encoders = X_encoders
        self.n_samples = n_samples
        self.n_features = n_features
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
            print(accuracy_score(self._predict(tree, X_test, y_test), y_test))
            print(self._prune(tree.root, tree, X_test, y_test))
            print(accuracy_score(self._predict(tree, X_test), y_test))

    def _build(self, tree, examples_idx, features_idx, depth=0):
        items, counts = unique(self.y[examples_idx])
        if (features_idx.size == 0
                or items.size == 1
                or examples_idx.size < self.min_samples_split
                or depth >= self.max_depth):
            node = self._class_node(items, counts)
            tree.classification_nodes.append(node)
            return node

        calc_record = self.splitter.calc(examples_idx, features_idx)

        if calc_record is None:
            node = self._class_node(items, counts)
            tree.classification_nodes.append(node)
            return node

        split_records = self.splitter.split(examples_idx, calc_record)

        features_idx = np.compress(calc_record.alive_features, features_idx)
        if not self.is_repeating:
            features_idx = np.delete(features_idx,
                                     np.where(features_idx ==
                                              calc_record.feature_idx))
        root = Node(calc_record.feature_idx,
                    is_feature=True,
                    details=calc_record)
        tree.feature_nodes.append(root)
        for record in split_records:
            if record.size == 0:
                node = self._class_node(items, counts)
                tree.classification_nodes.append(node)
                root.add_child(node, record)
            else:
                root.add_child(self._build(tree, record.bag,
                               features_idx, depth+1),
                               record)
        return root

    def _class_node(self, items, counts):
        classification = items[np.argmax(counts)]
        c_name = self.y_encoder.single_inv_transform(classification)
        node = Node(c_name)
        return node

    def _prune(self, node, tree, X_test, y_test):
        if node.is_feature:
            predicts = []
            n_children_incorrect = 0
            for n, _ in node.children:
                self._prune(n, tree, X_test, y_test)
                predicts += n.correct_predicts
                predicts += n.incorrect_predicts
                n_children_incorrect += len(n.incorrect_predicts)
                n.correct_predicts = []
                n.incorrect_predicts = []
            children_error_rate = 0
            node_error_rate = 0
            if len(predicts) != 0:
                children_error_rate = n_children_incorrect / float(len(predicts))
                max_predict_class = max(predicts, key=predicts.count)
                for predict in predicts:
                    if predict == max_predict_class:
                        node.correct_predicts.append(predict)
                    else:
                        node.incorrect_predicts.append(predict)

                node_error_rate = (len(node.incorrect_predicts)
                                   / float(len(predicts)))

            if len(predicts) != 0 and node_error_rate <= children_error_rate:
                decoded_class = (self
                                 .y_encoder
                                 .single_inv_transform(max_predict_class))
                node.is_feature = False
                node.value = decoded_class
                node.children = []

    def _predict(self, tree, X, y=None):
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
                node.add_predict_result(y[i])
        return ret
