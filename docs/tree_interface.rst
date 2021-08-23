Low-level Tree Interface
========================

The tree interface mimics that of ``sklearn.tree._tree.Tree``. In order to get this
level of detail, the argument ``enable_tree_details`` must be set to ``True`` on
estimators prior to fitting. Since ranger doesn't track this level of detail
on its trees, we perform extra (slow) operations in Python to provide them.

These operations include
* determining which leaf node corresponds to each training sample for every tree
* determining the sum of the weights of the training samples at each leaf node
* determining the weighted average prediction value of each leaf node

These extra calculations deliver ``n_node_values``, ``weighted_n_node_values``,
and ``value`` attributes of the ``Tree`` class.

.. note::

    Since we don't have direct access to the sub-sampled training sets used in building
    each of the trees, we determine the above values using the full training set.


.. autoclass:: skranger.tree._tree.Tree
    :members:
    :inherited-members: