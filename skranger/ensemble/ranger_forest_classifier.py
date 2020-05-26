"""Scikit-learn wrapper for ranger."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skranger.base import RangerValidationMixin
from skranger.ensemble import ranger


class RangerForestClassifier(RangerValidationMixin, ClassifierMixin, BaseEstimator):
    """Ranger Random Forest Probability/Classification implementation for sci-kit learn.

    Provides a sklearn classifier interface to the Ranger C++ library using Cython. The
    argument names to the constructor are similar to the C++ library and accompanied R
    package for familiarity.

    :param int num_trees: The number of tree classifiers to train
    :param bool verbose: Enable ranger's verbose logging
    :param int/callable mtry: The number of variables to split on each node. When a
        callable is passed, the function must accept a single parameter which is the
        number of features passed, and return some value between 1 and the number of
        features.
    :param str importance: One of one of ``none``, ``impurity``, ``impurity_corrected``,
        ``permutation``.
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
    :param list always_split_variables:  Variables which should always be selected for
        splitting.
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
    :param bool holdout: Hold-out all samples with case weight 0 and use these for
        variable importance and prediction error.
    :param bool oob_error: Whether to calculate out-of-bag prediction error.
    :param int num_threads: The number of threads. Default is number of CPU cores.
    :param bool save_memory: Save memory at the cost of speed growing trees.
    :param int seed: Random seed value.

    :ivar list classes\_: The class labels determined from the fit input ``y``.
    :ivar int n_classes\_: The number of unique class labels from the fit input ``y``.
    :ivar int n_features\_: The number of features (columns) from the fit input ``X``.
    :ivar list variable_names\_: Names for the features of the fit input ``X``.
    :ivar dict ranger_forest\_: The returned result object from calling C++ ranger.
    :ivar int mtry\_: The mtry value as determined if ``mtry`` is callable, otherwise
        it is the same as ``mtry``.
    :ivar list sample_fraction\_: The sample fraction determined by input validation
    :ivar list regularization_factor\_: The regularization factors determined by input
        validation.
    :ivar list unordered_variable_names\_: The unordered variable names determined by
        input validation.
    :ivar int split_rule\_: The split rule integer corresponding to ranger enum
        ``SplitRule``.
    :ivar bool use_regularization_factor\_: Input validation determined bool for using
        regularization factor input parameter.
    :ivar int importance_mode\_: The importance mode integer corresponding to ranger
        enum ``ImportanceMode``.
    """

    def __init__(
        self,
        num_trees=100,
        verbose=False,
        mtry=0,
        importance="none",
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
        """Fit the ranger random forest using training data.

        :param np.ndarray X: training input features
        :param np.ndarray y: training input classes
        :param np.ndarray sample_weight: optional weights for input samples
        """
        self.tree_type_ = 9  # tree_type, TREE_PROBABILITY enables predict_proba

        # Check input
        X, y = check_X_y(X, y)
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
            self.tree_type_,
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray(np.atleast_2d(y).astype("float64")),
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
            bool(self.split_select_weights),  # use_split_select_weights
            self.always_split_variables or [],  # always_split_variable_names
            bool(self.always_split_variables),  # use_always_split_variable_names
            False,  # prediction_mode
            {},  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            self.replace,  # sample_with_replacement
            False,  # probability
            self.unordered_variable_names_,
            bool(self.unordered_variable_names_),  # use_unordered_variable_names
            self.save_memory,
            self.split_rule_,
            sample_weight or [],  # case_weights
            bool(sample_weight),  # use_case_weights
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
        """Predict classes from X.

        :param array2d X: predict input features
        """
        probas = self.predict_proba(X)
        return self.classes_.take(np.argmax(probas, axis=1), axis=0)

    def predict_proba(self, X):
        """Predict probabilities for classes from X.

        :param array2d X: predict input features
        """
        check_is_fitted(self)
        X = check_array(X)

        result = ranger.ranger(
            9,  # tree_type, TREE_PROBABILITY
            np.asfortranarray(X.astype("float64")),
            np.array([[]]),
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
            bool(self.split_select_weights),  # use_split_select_weights
            self.always_split_variables or [],  # always_split_variable_names
            bool(self.always_split_variables),  # use_always_split_variable_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            self.replace,  # sample_with_replacement
            False,  # probability
            self.unordered_variable_names_,
            bool(self.unordered_variable_names_),  # use_unordered_variable_names
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
            self.use_regularization_factor_,
            self.regularization_usedepth,
        )
        return np.atleast_2d(np.array(result["predictions"]))

    def predict_log_proba(self, X):
        """Predict log probabilities for classes from X.

        :param array2d X: predict input features
        """
        proba = self.predict_proba(X)
        return np.log(proba)
