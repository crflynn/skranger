import warnings
from typing import Iterable

import numpy as np
from sklearn.utils.validation import _check_sample_weight

from skranger import ranger


class RangerMixin:
    @property
    def criterion(self):
        """Compatibility alias for split rule."""
        return self.split_rule

    # region validation
    def _validate_parameters(self, X, y, sample_weights):
        """Validate ranger parameters and set defaults."""
        if hasattr(self, "n_jobs"):
            self.n_jobs_ = max(
                [self.n_jobs, 0]
            )  # sklearn convention is -1 for all, ranger is 0
        self._set_respect_categorical_features()
        self._evaluate_mtry(X.shape[1])
        self._set_importance_mode()
        self.sample_fraction_ = self.sample_fraction or [1.0 if self.replace else 0.632]
        if not isinstance(self.sample_fraction_, Iterable):
            self.sample_fraction_ = [self.sample_fraction_]
        self._check_inbag(sample_weights)
        self._check_set_regularization(X.shape[1])
        self._set_split_rule(y)

    def _evaluate_mtry(self, num_features):
        """Evaluate mtry if callable."""
        if callable(self.mtry):
            self.mtry_ = self.mtry(num_features)
            if self.mtry_ < 1 or self.mtry_ > num_features:
                raise ValueError(
                    "mtry function must evaluate to between 1 and number of features"
                )
        else:
            self.mtry_ = self.mtry
            if self.mtry_ < 0 or self.mtry_ > num_features:
                raise ValueError("mtry must be between 0 and number of features")

    def _set_split_rule(self, y):
        """Set split rule to enum value."""
        if hasattr(self, "save_memory"):
            if (
                self.split_rule == "extratrees"
                and self.respect_categorical_features_ == "partition"
                and self.save_memory
            ):
                raise ValueError(
                    "save memory is not possible with extratrees split rule and unordered predictors"
                )

        if self.num_random_splits > 1 and self.split_rule != "extratrees":
            raise ValueError(
                "random splits must be 1 when split rule is not extratrees"
            )

        if self.tree_type_ in (1, 9):  # classification/probability
            if self.split_rule == "gini":
                self.split_rule_ = 1  # ranger_.SplitRule.LOGRANK
            elif self.split_rule == "extratrees":
                self.split_rule_ = 5  # ranger_.SplitRule.EXTRATREES
            elif self.split_rule == "hellinger":
                if len(np.unique(y)) > 2:
                    raise ValueError(
                        "hellinger split rule can only be used in binary classification"
                    )
                self.split_rule_ = 7  # ranger_.SplitRule.HELLINGER
            else:
                raise ValueError(
                    "split rule must be either gini, extratrees, or hellinger"
                )

        elif self.tree_type_ == 3:  # regression
            if self.split_rule == "variance":
                self.split_rule_ = 1  # ranger_.SplitRule.LOGRANK
            elif self.split_rule == "extratrees":
                self.split_rule_ = 5  # ranger_.SplitRule.EXTRATREES
            elif self.split_rule == "maxstat":
                self.split_rule_ = 4  # ranger_.SplitRule.MAXSTAT
            elif self.split_rule == "beta":
                self.split_rule_ = 6  # ranger_.SplitRule.BETA
                if np.max(y) > 1 or np.max(y) < 0:
                    raise ValueError(
                        "Targets must be between 0 and 1 for beta splitrule"
                    )
            else:
                raise ValueError(
                    "split rule must be either variance, extratrees, maxstat or beta"
                )

        elif self.tree_type_ == 5:  # survival
            if self.split_rule == "logrank":
                self.split_rule_ = 1  # ranger_.SplitRule.LOGRANK
            elif self.split_rule == "extratrees":
                self.split_rule_ = 5  # ranger_.SplitRule.EXTRATREES
            elif self.split_rule == "C":
                self.split_rule_ = 2  # ranger_.SplitRule.AUC
            elif self.split_rule == "C_ignore_ties":
                self.split_rule_ = 3  # ranger_.SplitRule.AUC_IGNORE_TIES
            elif self.split_rule == "maxstat":
                self.split_rule_ = 4  # ranger_.SplitRule.MAXSTAT
            else:
                raise ValueError(
                    "split rule must be either logrank, extratrees, C or maxstat"
                )

    def _check_set_regularization(self, num_features):
        """Check, set the regularization factor to either [] or length num_features."""
        if self.regularization_factor is None:
            self.regularization_factor_ = []
            self.use_regularization_factor_ = False
            return

        if len(self.regularization_factor) > 0:
            if max(self.regularization_factor) > 1:
                raise ValueError("The regularization coefficients must be <= 1")
            if max(self.regularization_factor) <= 0:
                raise ValueError("The regularization coefficients must be > 0")
            if (
                len(self.regularization_factor) != 1
                and len(self.regularization_factor) != num_features
            ):
                raise ValueError(
                    "There must be either one 1 or (number of features) regularization coefficients"
                )
            if len(self.regularization_factor) == 1:
                self.regularization_factor_ = [
                    self.regularization_factor
                ] * num_features

        if all([r == 1 for r in self.regularization_factor]):
            self.regularization_factor_ = []
            self.use_regularization_factor_ = False
        else:
            if getattr(self, "n_jobs_", 1) != 1:
                warnings.warn("Parallelization cannot be used with regularization.")
                self.n_jobs_ = 1
            self.regularization_factor_ = self.regularization_factor
            self.use_regularization_factor_ = True

    def _set_importance_mode(self):
        """Set the importance mode based on ``importance`` and ``local_importance``."""
        # Note IMP_PERM_LIAW is unused
        if self.importance is None or self.importance == "none":
            self.importance_mode_ = 0  # ranger_.ImportanceMode.IMP_NONE
        elif self.importance == "impurity":
            self.importance_mode_ = 1  # ranger_.ImportanceMode.IMP_GINI
        elif (
            self.importance == "impurity_corrected"
            or self.importance == "impurity_unbiased"
        ):
            self.importance_mode_ = 5  # ranger_.ImportanceMode.IMP_GINI_CORRECTED
        elif self.importance == "permutation":
            if self.local_importance:
                self.importance_mode_ = 6  # ranger_.ImportanceMode.IMP_PERM_CASEWISE
            elif self.scale_permutation_importance:
                self.importance_mode_ = 2  # ranger_.ImportanceMode.IMP_PERM_BREIMAN
            else:
                self.importance_mode_ = 3  # ranger_.ImportanceMode.IMP_PERM_RAW
        else:
            raise ValueError("unkown importance mode")

    def _check_inbag(self, sample_weights):
        """Validate related input against ``inbag`` counts."""
        if self.inbag:
            if sample_weights is not None:
                raise ValueError("Cannot use inbag and sample_weights.")
            if len(self.sample_fraction_) > 1:
                raise ValueError("Cannot use class sampling and inbag.")
            if len(self.inbag) != getattr(self, "n_estimators", 1):
                raise ValueError("Size of inbag must be equal to n_estimators.")

    def _check_sample_weight(self, sample_weight, X):
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            use_sample_weight = True
            # ranger does additional rng on samples if weights are passed.
            # if the weights are ones, then we dont want that extra rng.
            if np.array_equal(np.unique(sample_weight), np.array([1.0])):
                sample_weight = []
                use_sample_weight = False
        else:
            sample_weight = []
            use_sample_weight = False
        return sample_weight, use_sample_weight

    def _check_split_select_weights(self, split_select_weights):
        if split_select_weights is not None and len(split_select_weights) > 0:
            split_select_weights = np.atleast_2d(split_select_weights).tolist()
            use_split_select_weights = True
        else:
            split_select_weights = [[]]
            use_split_select_weights = False
        return split_select_weights, use_split_select_weights

    def _check_always_split_features(self, always_split_features):
        if always_split_features is not None and len(always_split_features) > 0:
            always_split_features = [str(c).encode() for c in always_split_features]
            use_always_split_features = True
        else:
            always_split_features = []
            use_always_split_features = False
        return always_split_features, use_always_split_features

    def _set_respect_categorical_features(self):
        """Set ``respect_categorical_features`` based on ``split_rule``."""
        if self.respect_categorical_features is None:
            if self.split_rule == "extratrees":
                self.respect_categorical_features_ = "partition"
            else:
                self.respect_categorical_features_ = "ignore"
        else:
            self.respect_categorical_features_ = self.respect_categorical_features

    def _check_categorical_features(self, categorical_features):
        """Determine categorical feature names."""
        use_categorical_features = False
        if self.respect_categorical_features_ == "partition":
            if categorical_features is not None and len(categorical_features) > 0:
                categorical_features = [str(c).encode() for c in categorical_features]
                use_categorical_features = True
        elif (
            self.respect_categorical_features_ == "ignore"
            or self.respect_categorical_features_ == "order"
        ):
            categorical_features = []
            use_categorical_features = False
        else:
            raise ValueError(
                "respect ordered factors must be one of `partition`, `ignore` or `order`"
            )
        if categorical_features is None:
            categorical_features = []
        return categorical_features, use_categorical_features

    # endregion

    # region trees
    def _set_sample_weights(self, sample_weights):
        """Set leaf weights for access in ``Tree``."""
        weights = []
        for tree in sample_weights:
            tree_weights = []
            for idx, node in enumerate(tree):
                tree_weights.append(sum(node))
            weights.append(tree_weights)
        self.ranger_forest_["forest"]["leaf_weights"] = weights

    def _get_sample_values(self, values):
        """Map the leaf samples to corresponding values.

        Create a similarly structured set of lists by mapping the leaf sample indexes
        to corresponding values passed. Used for mapping leaf samples to target values
        or sample weights.
        """
        mapped_values = []
        for tree in self.ranger_forest_["forest"]["leaf_samples"]:
            mapped_values.append([])
            for node in tree:
                mapped_values[-1].append([values[idx] for idx in node])
        return mapped_values

    def _get_values(self, left, right, idx, values):
        """Recursively sum the child values through a tree."""
        left_values = (
            values[idx]
            if left[idx] == 0
            else self._get_values(left, right, left[idx], values)
        )
        right_values = (
            values[idx]
            if right[idx] == 0
            else self._get_values(left, right, right[idx], values)
        )
        values[idx] = left_values + right_values
        return values[idx]

    def _set_node_values(self, y, sample_weight):
        """Set the node values for ``Tree.value``."""
        forest_values = self._get_sample_values(y)
        forest_weights = self._get_sample_values(sample_weight)
        self._set_sample_weights(forest_weights)
        self.ranger_forest_["forest"]["node_values"] = []
        for idx in range(self.ranger_forest_["num_trees"]):
            left = self.ranger_forest_["forest"]["child_node_ids"][idx][0]
            right = self.ranger_forest_["forest"]["child_node_ids"][idx][1]
            root = 0
            values = forest_values[idx]
            self._get_values(
                left, right, root, values,
            )
            weights = forest_weights[idx]
            self._get_values(
                left, right, root, weights,
            )
            values = [
                np.average(v, weights=w, axis=0) if v else np.nan
                for v, w in zip(values, weights)
            ]
            self.ranger_forest_["forest"]["node_values"].append(values)

    def _set_n_classes(self):
        """Set num classes for ``Tree.n_classes``."""
        # for accessing in Tree
        self.ranger_forest_["n_classes"] = getattr(self, "n_classes_", 1)

    def _set_leaf_samples(self, terminal_nodes):
        """Set the leaf samples using the terminal nodes.

        Collect all of the record indexes that fall into each terminal node. The
        resulting ``leaf_samples`` value will be a list of lists of lists. The outer
        list is a collection of tree lists. The tree lists are collections of node
        lists. The node lists contain record indexes for terminal nodes only. The index
        of the node lists correspond to the node index in the respective tree.
        """
        leaf_samples = []
        for tree_idx, tree in enumerate(terminal_nodes.T):
            n_nodes = len(self.ranger_forest_["forest"]["child_node_ids"][tree_idx][0])
            tree_leaf_samples = [[] for _ in range(n_nodes)]
            for record_idx, terminal_node in enumerate(tree):
                tree_leaf_samples[terminal_node].append(record_idx)
            leaf_samples.append(tree_leaf_samples)
        self.ranger_forest_["forest"]["leaf_samples"] = leaf_samples

    def _get_terminal_node_forest(self, X):
        """Get a terminal node forest for X.

        Uses a trained forest to return the terminal node ids of each record of ``X``
        for each tree. Returns a dictionary.

        The returned value of key ``predictions`` will hold a list of
        lists. The inner list is the list of terminal nodes of each tree for a record.
        The outer list entries correspond to each record of ``X``.

        :param array2d X: prediction input features
        """
        # many fields defaulted here which are unused
        forest = ranger.ranger(
            self.tree_type_,
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray([[]]),
            self.feature_names_,  # variable_names
            0,  # m_try
            getattr(self, "n_estimators", 1),  # num_trees
            self.verbose,
            self.seed,
            getattr(self, "n_jobs_", 1),  # num_threads
            False,  # write_forest
            0,  # importance_mode
            0,  # min_node_size
            [],  # split_select_weights
            False,  # use_split_select_weights
            [],  # always_split_feature_names
            False,  # use_always_split_feature_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            True,  # sample_with_replacement
            False,  # probability
            [],  # unordered_feature_names
            False,  # use_unordered_features
            False,  # save_memory
            1,  # split_rule
            [],  # case_weights
            False,  # use_case_weights
            {},  # class_weights
            False,  # predict_all
            self.keep_inbag,
            [1],  # sample_fraction
            0,  # alpha
            0,  # minprop
            self.holdout,
            2,  # prediction_type (terminal nodes)
            1,  # num_random_splits
            False,  # oob_error
            0,  # max_depth
            [],  # inbag
            False,  # use_inbag
            [],  # regularization_factor_
            False,  # use_regularization_factor_
            False,  # regularization_usedepth
        )
        return forest

    # endregion
