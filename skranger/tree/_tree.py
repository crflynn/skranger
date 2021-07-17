import numpy as np
from sklearn.tree._tree import csr_matrix


class Tree:
    """The low-level tree interface.

    Tree objects can be accessed using the ``tree_`` attribute on fitted
    decision tree estimators. Instances of ``Tree`` provide methods and
    properties describing the underlying structure and attributes of the
    tree.
    """

    def __init__(self, ranger_forest):
        self.ranger_forest = ranger_forest

    @property
    def node_count(self):
        """The quantity of nodes in the tree."""
        return len(self.children_left)

    @property
    def capacity(self):
        """The capacity of the node array."""
        return len(self.children_left)

    @property
    def children_left(self):
        """Left children nodes."""
        # sklearn uses -1, ranger uses 0
        return np.array(
            [
                -1 if n == 0 else n
                for n in self.ranger_forest["forest"]["child_node_ids"][0][0]
            ]
        )

    @property
    def children_right(self):
        """Right children nodes."""
        # sklearn uses -1, ranger uses 0
        return np.array(
            [
                -1 if n == 0 else n
                for n in self.ranger_forest["forest"]["child_node_ids"][0][1]
            ]
        )

    @property
    def n_outputs(self):
        """The quantity of outputs of the tree."""
        # single output only
        return 1

    @property
    def n_classes(self):
        """The quantity of classes."""
        return np.array([self.ranger_forest["n_classes"]])

    def get_depth(self):
        """Calculate the maximum depth of the tree."""
        left = self.children_left
        right = self.children_right
        root_node = 0
        return self._get_depth(left, right, root_node)

    def _get_depth(self, left, right, idx):
        if left[idx] == -1:
            return 0
        return 1 + max(
            self._get_depth(left, right, left[idx]),
            self._get_depth(left, right, right[idx]),
        )

    def get_n_leaves(self):
        """Calculate the number of leaves of the tree."""
        left = self.children_left
        right = self.children_right
        root_node = 0
        return self._get_n_leaves(left, right, root_node)

    def _get_n_leaves(self, left, right, idx):
        if left[idx] == -1:
            return 1
        return self._get_n_leaves(left, right, left[idx]) + self._get_n_leaves(
            left, right, right[idx]
        )

    def apply(self, X):
        """Calculate the leaf index for each sample.

        :param array2d X: training input features
        """
        return np.apply_along_axis(self._apply, 1, X)

    def _apply(self, x, idx=None):
        if idx is None:
            idx = 0
            return self._apply(x, idx)
        if self.children_left[idx] == -1:
            return idx
        varid = self.feature[idx]
        val = self.threshold[idx]
        if x[varid] <= val:
            idx = self.children_left[idx]
        else:
            idx = self.children_right[idx]
        return self._apply(x, idx)

    def decision_path(self, X):
        """Calculate the decision path through the tree for each sample.

        :param array2d X: training input features
        """
        if hasattr(X, "values"):  # pd.Dataframe
            Xvalues = X.values
        else:
            Xvalues = X
        paths = [self._decision_path(x) for x in Xvalues]
        rows = [np.ones(len(p), dtype=int) * idx for idx, p in enumerate(paths)]
        rows = np.concatenate(rows, axis=0)
        cols = np.concatenate(paths, axis=0)
        data = np.ones(len(rows), dtype=int)
        return csr_matrix((data, (rows, cols)))

    def _decision_path(self, x, idx=None):
        if idx is None:
            idx = 0
            return [idx] + self._decision_path(x, idx)
        if self.children_left[idx] == -1:
            return []
        varid = self.feature[idx]
        val = self.threshold[idx]
        if x[varid] <= val:
            idx = self.children_left[idx]
        else:
            idx = self.children_right[idx]
        return [idx] + self._decision_path(x, idx)

    @property
    def max_depth(self):
        """Max depth of the tree."""
        return self.get_depth()

    @property
    def feature(self):
        """Variables on which nodes are split."""
        # sklearn uses -2, ranger uses 0
        return np.array(
            [
                -2 if v == 0 else v
                for v in self.ranger_forest["forest"]["split_var_ids"][0]
            ]
        )

    @property
    def threshold(self):
        """Threshold values on which nodes are split."""
        # sklearn uses -2, ranger uses 0
        return np.array(
            [
                -2 if v == 0 else v
                for v in self.ranger_forest["forest"]["split_values"][0]
            ]
        )

    @property
    def n_node_samples(self):
        """The number of samples reaching each node."""
        n_samples = [
            len(node) if node else 0
            for node in self.ranger_forest["forest"]["leaf_samples"][0]
        ]
        self._get_n_node_samples(self.children_left, self.children_right, 0, n_samples)
        return np.array(n_samples)

    def _get_n_node_samples(self, left, right, idx, n_samples):
        left_n_node_samples = (
            n_samples[idx]
            if left[idx] == -1
            else self._get_n_node_samples(left, right, left[idx], n_samples)
        )
        right_n_node_samples = (
            n_samples[idx]
            if right[idx] == -1
            else self._get_n_node_samples(left, right, right[idx], n_samples)
        )
        n_samples[idx] = left_n_node_samples + right_n_node_samples
        return n_samples[idx]

    @property
    def weighted_n_node_samples(self):
        """The sum of the weights of the samples reaching each node."""
        weighted_n_samples = self.ranger_forest["forest"]["leaf_weights"][0].copy()
        self._get_n_node_samples(
            self.children_left, self.children_right, 0, weighted_n_samples,
        )
        return np.array(weighted_n_samples)

    @property
    def value(self):
        """The constant prediction value of each node."""
        values = self.ranger_forest["forest"]["node_values"][0]
        return np.reshape(values, (len(values), 1, 1))
