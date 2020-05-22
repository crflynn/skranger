"""Scikit-learn wrapper for ranger."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import _check_sample_weight

import skranger.ranger as ranger


class RangerForestClassifier(ClassifierMixin, BaseEstimator):
    """Ranger Random Forest implementation for sci-kit learn.

    :param int num_trees: The number of tree classifiers to train
    :param bool verbose: Enable ranger's verbose logging
    :param int/callable mtry: The number of variables to split on each node. When a
        callable is passed, the function must accept a single parameter which is the
        number of features passed, and return some value between 1 and the number of
        features.
    :param str importance: One of one of ``none``, ``impurity``, ``impurity_corrected``,
        ``permutation``.
    :param bool probability: Grow a probability forest as in Malley et al. (2012)
    :param int min_node_size: The minimal node size.
    :param int max_depth: The maximal tree depth; 0 means unlimited.
    :param bool replace: Sample with replacement.
    :param sample_fraction: The fraction of observations to sample. The default is 1
        when sampling with replacement, and 0.632 otherwise. This can be a vector of
        class specific values.
    :param list class_weights: Weights for the outcome classes.
    :param str split_rule: One of ``gini``, ``extratrees``, ``hellinger``;
        default ``gini``.
    :param int num_random_splits: The number of trees for the ``extratrees`` splitrule.
    :param list split_select_weights: Vector of weights between 0 and 1 of probabilities
        to select variables for splitting.
    :param list always_split_variables:  TODO
    :param str respect_unordered_factors: One of ``ignore``, ``order``, ``partition``.
        The default is ``partition`` for the ``extratrees`` splitrule, otherwise the
        default is ``ignore``.
    :param bool scale_permutation_importance: For ``permutation`` importance,
        scale permutation importance by standard error as in (Breiman 2001).
    :param bool local_importance: For ``permutation`` importance, calculate and
        return local importance values as (Breiman 2001).
    :param list regularization_factor: A vector of regularization factors for the
        features.
    :param bool regularization_usedepth: Whether to consider depth in regularization.
    :param bool holdout: Hold-out all samples with case weight 0 and use these for variable
        importance and prediction error.
    :param bool oob_error: Whether to calculate OOB prediction error.
    :param int num_threads: The number of threads. Default is number of CPU cores.
    :param bool save_memory: Save memory at the cost of speed growing trees.
    :param int seed: Random seed value.
    """

    def __init__(
        self,
        num_trees=500,
        verbose=False,
        mtry=0,
        importance="none",
        probability=False,
        min_node_size=0,
        max_depth=0,
        replace=True,
        sample_fraction=None,
        class_weights=None,
        split_rule="gini",
        num_random_splits=1,
        split_select_weights=None,
        always_split_variables=None,
        respect_unordered_factors=None,
        scale_permutation_importance=False,
        local_importance=False,
        regularization_factor=None,
        regularization_usedepth=False,
        holdout=False,
        oob_error=False,
        num_threads=0,
        save_memory=False,
        seed=42,
    ):
        self.num_trees = num_trees
        self.verbose = verbose
        self.mtry = mtry
        self.importance = importance
        self.probability = probability
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.replace = replace
        self.sample_fraction = sample_fraction
        self.class_weights = class_weights
        self.split_rule = split_rule
        self.num_random_splits = num_random_splits
        self.split_select_weights = split_select_weights
        self.always_split_variables = always_split_variables
        self.respect_unordered_factors = respect_unordered_factors
        self.scale_permutation_importance = scale_permutation_importance
        self.local_importance = local_importance
        self.regularization_factor = regularization_factor
        self.regularization_usedepth = regularization_usedepth
        self.holdout = holdout
        self.oob_error = oob_error
        self.num_threads = num_threads
        self.save_memory = save_memory
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Check the init parameters
        self._validate_parameters(X, y)

        # Map classes to indices
        y = y.copy()
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        # Store X info
        self.n_features_ = X.shape[1]
        self.variable_names_ = [str(r).encode() for r in range(X.shape[1])]

        # Fit the forest
        self.ranger_forest_ = ranger.ranger(
            9,  # tree_type, TREE_PROBABILITY enables predict_proba
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray(y.astype("float64")),
            self.variable_names_,
            self.mtry,
            self.num_trees,
            self.verbose,
            self.seed,
            self.num_threads,
            True,  # write_forest
            self.importance_mode_,
            self.min_node_size,
            self.split_select_weights or [],
            False,  # use_split_select_weights
            self.always_split_variables or [],  # always_split_variable_names
            False,  # use_always_split_variable_names
            False,  # prediction_mode
            {},  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            self.replace,  # sample_with_replacement
            False,  # probability
            self.unordered_variable_names_,
            False,  # use_unordered_variable_names
            self.save_memory,
            self.split_rule_,
            sample_weight or [],  # case_weights
            False,  # use_case_weights
            self.class_weights or [],
            False,  # predict_all
            False,  # keep_inbag
            self.sample_fraction_,
            0.5,  # alpha
            0.1,  # minprop
            self.holdout,
            1,  # prediction_type
            self.num_random_splits,
            False,  # use_sparse_data
            self.order_snps_,
            self.oob_error,
            self.max_depth,
            [],  # inbag
            False,  # use_inbag
            self.regularization_factor_,
            False,  # use_regularization_factor
            self.regularization_usedepth,
        )
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_.take(np.argmax(probas, axis=1), axis=0)

    def predict_proba(self, X):
        result = ranger.ranger(
            9,  # tree_type, TREE_PROBABILITY for class probabilities
            np.asfortranarray(X.astype("float64")),
            np.array([]),
            self.variable_names_,
            self.mtry,
            self.num_trees,
            self.verbose,
            self.seed,
            self.num_threads,
            False,  # write_forest
            self.importance_mode_,
            self.min_node_size,
            self.split_select_weights or [],
            False,  # use_split_select_weights
            self.always_split_variables or [],  # always_split_variable_names
            False,  # use_always_split_variable_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            self.replace,  # sample_with_replacement
            False,  # probability
            self.unordered_variable_names_,
            False,  # use_unordered_variable_names
            self.save_memory,
            self.split_rule_,
            [],  # case_weights
            False,  # use_case_weights
            self.class_weights or [],
            False,  # predict_all
            False,  # keep_inbag
            self.sample_fraction_,
            0.5,  # alpha
            0.1,  # minprop
            self.holdout,
            1,  # prediction_type
            self.num_random_splits,
            False,  # use_sparse_data
            self.order_snps_,
            self.oob_error,
            self.max_depth,
            [],  # inbag
            False,  # use_inbag
            self.regularization_factor_,
            False,  # use_regularization_factor
            self.regularization_usedepth,
        )
        return np.array(result["predictions"])

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(proba)

    def _validate_parameters(self, X, y):
        check_X_y(X, y)
        self._set_respect_unordered_factors()
        # TODO order mode recoding
        self._evaluate_mtry(X.shape[1])
        self._set_importance_mode()
        self._check_set_regularization(X.shape[1])

        self.sample_fraction_ = self.sample_fraction or [1.0 if self.replace else 0.632]

        self.regularization_factor_ = self.regularization_factor or [1.0] * X.shape[1]

        self._set_split_rule()
        self.order_snps_ = self.respect_unordered_factors == "order"

        self._set_unordered_variable_names()

    def _set_unordered_variable_names(self):
        if self.respect_unordered_factors == "partition":
            # TODO check which ones are ordered and factored
            pass
        elif self.respect_unordered_factors == "ignore" or self.respect_unordered_factors == "order":
            self.unordered_variable_names_ = []
        else:
            raise ValueError("respect ordered factors must be one of `partition`, `ignore` or `order`")

    def _evaluate_mtry(self, num_features):
        if callable(self.mtry):
            self.mtry_ = self.mtry(num_features)
            if self.mtry_ < 1 or self.mtry_ > num_features:
                raise ValueError("mtry function must evaluate to between 1 and number of features")
        else:
            self.mtry_ = self.mtry
            if self.mtry_ < 0 or self.mtry_ > num_features:
                raise ValueError("mtry must be between 0 and number of features")

    def _set_split_rule(self):
        if self.split_rule == "gini":
            self.split_rule_ = 1  # ranger_.SplitRule.LOGRANK
        elif self.split_rule == "extratrees":
            self.split_rule_ = 5  # ranger_.SplitRule.EXTRATREES
        elif self.split_rule == "hellinger":
            self.split_rule_ = 7  # ranger_.SplitRule.HELLINGER
        else:
            raise ValueError("split rule must be either gini, extratrees, or hellinger")

        if self.split_rule == "extratrees" and self.respect_unordered_factors == "partition" and self.save_memory:
            raise ValueError("save memory is not possible with extratrees split rule and unordered predictors")

        if self.num_random_splits > 1 and self.split_rule != "extratrees":
            raise ValueError("random splits must be 1 when split rule is not extratrees")

    def _set_respect_unordered_factors(self):
        if self.respect_unordered_factors is None:
            if self.split_rule == "extratrees":
                self.respect_unordered_factors = "partition"
            else:
                self.respect_unordered_factors = "ignore"

    def _check_set_regularization(self, num_features):
        """Check, set the regularization factor to either [] or length num_features"""
        if self.regularization_factor is None:
            self.regularization_factor = []
            return
        if len(self.regularization_factor) > 0:
            if max(self.regularization_factor) > 1:
                raise ValueError("The regularization coefficients must be <= 1")
            if max(self.regularization_factor) <= 0:
                raise ValueError("The regularization coefficients must be > 0")
            if len(self.regularization_factor) != 1 and len(self.regularization_factor) != num_features:
                raise ValueError("There must be either one 1 or (number of features) regularization coefficients")
            if len(self.regularization_factor) == 1:
                self.regularization_factor = self.regularization_factor * num_features
        if all([r == 1 for r in self.regularization_factor]):
            self.regularization_factor = []
            self.use_regularization_factor_ = False
        else:
            if self.num_threads != 1:
                self.num_threads = 1
                # TODO Warn parallelization cannot be used with regularization
            self.use_regularization_factor_ = True

    def _set_importance_mode(self):
        # Note IMP_PERM_LIAW is unused
        if self.importance is None or self.importance == "none":
            self.importance_mode_ = 0  # ranger_.ImportanceMode.IMP_NONE
        elif self.importance == "impurity":
            self.importance_mode_ = 1  # ranger_.ImportanceMode.IMP_GINI
        elif self.importance == "impurity_corrected" or self.importance == "impurity_unbiased":
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
