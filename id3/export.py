from sklearn.externals import six
import pydotplus

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


def export_pdf(decision_tree, out_file='tree.pdf', feature_names=None, class_names=None):
    dot_tree = export_graphviz(decision_tree)
    graph = pydotplus.graph_from_dot_data(dot_tree.to_string())
    graph.write_pdf(out_file)

def export_graphviz(decision_tree, out_file=DotTree(), feature_names=None, class_names=None):
    """Export a decision tree in DOT format.
    
    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default='tree.dot')
        Handle or name of the output file. If ``None``, the result is
        returned as a string. This will the default from version 0.20.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    class_names : list of strings, bool or None, optional (default=None)
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    Returns
    -------
    dot_data : string
        String representation of the input tree in GraphViz dot format.
    """
    ranks = {}
    node_ids = []
    max_depth = 500
    def _recurse_tree(node, node_id=0, edge=None, parent=None, depth=0):
        #DOT language can't handle ``-``, use ``_`` instead
        depth += 1
        node_ids.append(_get_next_id())
        if max_depth is None or depth <= max_depth:
            out_file.write(_node_to_dot(node, node_id, parent, edge, depth))
            for child, edge in node.children:
                _recurse_tree(child, _get_next_id(), edge, node_id, depth)

    def _get_next_id():
        if len(node_ids) == 0:
            return 0
        else:
            return node_ids[-1] + 1
    
    def _node_to_dot(node, n_id=0, parent=None, edge=None, depth=0):
        """Get  a Node objects representation in dot format.

        """
        node_repr = []
        if str(depth) not in ranks:
            ranks[str(depth)] = []
        ranks[str(depth)].append(str(n_id))
        
        node_repr.append('\"{}\" [shape=box, style=filled, label=\"{}\", weight={}]\n'.format(n_id, node.value, depth))
        if parent != None:
            node_repr.append('{} -> {} [ label = "{}"];\n'.format(parent, n_id, edge, depth))
        res = "".join(node_repr)
        return res
            

    
    if not isinstance(out_file, DotTree) and six.PY3:
        out_file = open(out_file, 'w', encoding='utf8')
    elif not isinstance(out_file, DotTree):
        out_file = open(out_file, 'wb')

    out_file.write('digraph ID3_Tree {\n')
    _recurse_tree(decision_tree)

    for rank in sorted(ranks):
        out_file.write("{rank=same; ")
        for r in ranks[rank]:
            out_file.write(str(r) + ";")
        out_file.write("};\n")
    out_file.write("}")
    out_file.close()
    return out_file

