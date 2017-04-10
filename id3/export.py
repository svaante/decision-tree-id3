from sklearn.externals import six

def export_graphviz(decision_tree, out_file='tree.dot', feature_names=None, class_names=None):
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
    max_depth = 500
    def _recurse_tree(node, edge=None, parent=None, depth=0):
        if max_depth is None or depth <= max_depth:
            out_file.write(_node_to_dot(node, parent, edge))
            for child, edge in node.children:
                _recurse_tree(child, edge, node)

    def _node_to_dot(node, parent=None, edge=None):
        """Get  a Node objects representation in dot format.

        """
        node_repr = []
        
        if parent != None:
            node_repr.append('{} -> {} [ label = "{}" ];\n'.format(parent.name, node.name, edge))
        res = "".join(node_repr)
        return res
            

    
    if six.PY3:
        out_file = open(out_file, 'w', encoding='utf8')
    else:
        out_file = open(out_file, 'wb')

    out_file.write('digraph ID3_Tree {\n')
    out_file.write('node [shape=box];\n')
    decision_tree.print_tree()
    _recurse_tree(decision_tree)
    out_file.write("}")
    out_file.close()
