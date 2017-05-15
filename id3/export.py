from sklearn.externals import six
from .splitter import SplitRecord, CalcRecord
import numpy as np


class DotTree():

    def __init__(self):
        self.dot_tree = ""
        self.closed = False

    def write(self, content):
        if not self.closed:
            self.dot_tree += content

    def close(self):
        self.closed = True

    def to_string(self):
        return self.dot_tree


def _extract_class_count(node):
    if node.item_count is not None:
        items, counts = node.item_count
        if counts.size == 0:
            return "(0)"
        elif counts.shape[0] == 1:
            return "({})".format(counts[0])
        else:
            max_count = np.max(counts)
            incorrect_count = np.sum(counts) - max_count
            return "({}/{})".format(max_count, incorrect_count)
    else:
        return ""


def _extract_edge_value(tree, edge):
    ft_idx = edge.calc_record.feature_idx
    split_type = edge.calc_record.split_type
    val = edge.value_encoded
    pivot = edge.calc_record.pivot
    if split_type is CalcRecord.NUM:
        if val == SplitRecord.GREATER:
            return ">{0:.2f}".format(pivot)
        else:
            return "<={0:.2f}".format(pivot)
    elif tree.X_encoders is not None:
        value = tree.X_encoders[ft_idx].single_inv_transform(val)
        if isinstance(value, np.bytes_):
            return value.decode('UTF-8')
        else:
            return value
    else:
        return val


def export_text(decision_tree, feature_names=None):
    """Export a decision tree in WEKA like string format.
    Parameters
    ----------
    decision_tree : decision tree classifier
    feature_names : list of strings, optional (default=None)
        Names of each of the features.
    Returns
    -------
    ret : string
    """
    max_depth = 500

    def build_string(node, indent, depth):
        ret = ''
        if node is None or depth > max_depth:
            return ''
        if node.is_feature:
            ret += '\n'
            template = '|   ' * indent
            if feature_names is None:
                template += str(node.details.feature_idx)
            else:
                template += feature_names[node.details.feature_idx]
            template += ' {}'
            for child in node.children:
                edge_value = _extract_edge_value(decision_tree, child[1])
                ret += template.format(edge_value)
                ret += build_string(child[0], indent + 1, depth + 1)
        else:
            value = decision_tree.y_encoder.single_inv_transform(node.value)
            if isinstance(value, np.bytes_):
                value = value.decode('UTF-8')
            ret += ': {} {} \n'.format(value, _extract_class_count(node))
        return ret
    return build_string(decision_tree.root, 0, 0)


def export_graphviz(decision_tree, out_file=DotTree(), feature_names=None,
                    extensive=False):
    """Export a decision tree in DOT format.
    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported,
    graphical renderings can be generated using, for example:
        $ dot -Tpdf tree.dot -o tree.pdf    (PDF format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)
    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.
    out_file : string, optional (default=DotTree)
        Name of the output file. If ``None``, the result is
        returned as a string.
    feature_names : list of strings, optional (default=None)
        Names of each of the features.
    extensive : displays aditional information, optional (default=False)

    Returns
    -------
    dot_data : string
        String representation of the input tree in GraphViz dot format.
    """
    ranks = {}
    node_ids = []
    max_depth = 500

    def _recurse_tree(node, node_id=0, edge=None, parent=None, depth=0):
        depth += 1
        if depth > max_depth:
            return

        node_ids.append(_get_next_id())
        out_file.write(_node_to_dot(node, node_id, parent, edge, depth))
        for child, edge in node.children:
            _recurse_tree(child, _get_next_id(), edge, node_id, depth)

    def _get_next_id():
        if len(node_ids) == 0:
            return 0
        else:
            return node_ids[-1] + 1

    def _node_to_dot(node, n_id=0, parent=None, edge=None, depth=0):
        """Get a Node objects representation in dot format.
        """
        node_repr = []
        if str(depth) not in ranks:
            ranks[str(depth)] = []
        ranks[str(depth)].append(str(n_id))

        node_repr.append(('\"{}\" [shape=box, style=filled, label=\"{}\", '
                          'weight={}]\n')
                         .format(n_id, _extract_node_info(node), depth))
        if parent is not None:
            node_repr.append(('{} -> {} [ label = "{}"];\n')
                             .format(parent,
                                     n_id,
                                     _extract_edge_value(decision_tree,
                                                         edge)))
        res = "".join(node_repr)
        return res

    def _extract_node_info(node):
        result = ""
        value = ""

        if feature_names is not None and node.is_feature:
            value = str(feature_names[node.value])
        elif not node.is_feature:
            value = (decision_tree.y_encoder
                     .single_inv_transform(node.value))
        else:
            value = node.value

        if isinstance(value, np.bytes_):
            value = value.decode('UTF-8')
        result += str(value) + "\n"
        if node.is_feature and extensive:
            class_counts = node.details.class_counts
            dominant_class = class_counts[np.argmax(class_counts[:, 1]), :]
            result += ("Info: {0:.2f}\n"
                       .format(node.details.info))
            result += ("Entropy: {0:.2f}\n"
                       .format(node.details.entropy))
            result += "Dominant class: {}\n".format(dominant_class)
        if not node.is_feature:
            result += _extract_class_count(node) + "\n"
        return result

    if not isinstance(out_file, DotTree) and six.PY3:
        out_file = open(out_file, 'w', encoding='utf8')
    elif not isinstance(out_file, DotTree):
        out_file = open(out_file, 'wb')

    out_file.write('digraph ID3_Tree {\n')
    _recurse_tree(decision_tree.root)

    for rank in sorted(ranks):
        out_file.write("{rank=same; ")
        for r in ranks[rank]:
            out_file.write(str(r) + ";")
        out_file.write("};\n")
    out_file.write("}")
    out_file.close()
    return out_file
