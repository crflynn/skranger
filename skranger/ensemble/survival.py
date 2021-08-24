"""Scikit-learn wrapper for ranger survival."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skranger import ranger
from skranger.ensemble.base import BaseRangerForest
from skranger.tree import RangerTreeSurvival


class RangerForestSurvival(BaseRangerForest, BaseEstimator):
    r"""Ranger Random Forest Survival implementation for sci-kit survival.

    Provides a sksurv interface to the Ranger C++ library using Cython.

    :param int n_estimators: The number of tree classifiers to train
    :param bool verbose: Enable ranger's verbose logging
    :param int/callable mtry: The number of features to split on each node. When a
        callable is passed, the function must accept a single parameter which is the
        number of features passed, and return some value between 1 and the number of
        features.
    :param str importance: One of one of ``none``, ``impurity``, ``impurity_corrected``,
        ``permutation``.
    :param int min_node_size: The minimal node size.
    :param int max_depth: The maximal tree depth; 0 means unlimited.
    :param bool replace: Sample with replacement.
    :param float sample_fraction: The fraction of observations to sample. The default
        is 1 when sampling with replacement, and 0.632 otherwise.
    :param bool keep_inbag: If true, save how often observations are in-bag in each
        tree. These will be stored in the ``ranger_forest_`` attribute under the key
        ``"inbag_counts"``.
    :param list inbag: A list of size ``n_estimators``, containing inbag counts for each
        observation. Can be used for stratified sampling.
    :param str split_rule: One of ``logrank``, ``extratrees``, ``C``, or ``maxstat``,
        default ``logrank``.
    :param int num_random_splits: The number of random splits to consider for the
        ``extratrees`` splitrule.
    :param float alpha: Significance threshold to allow splitting for the ``maxstat``
        split rule.
    :param float minprop: Lower quantile of covariate distribution to be considered for
        splitting for ``maxstat`` split rule.
    :param str respect_categorical_features: One of ``ignore``, ``order``, ``partition``.
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
        feature importance and prediction error.
    :param bool oob_error: Whether to calculate out-of-bag prediction error.
    :param int n_jobs: The number of threads. Default is number of CPU cores.
    :param int seed: Random seed value.
    :param bool enable_tree_details: When ``True``, perform additional calculations
        for detailing the underlying decision trees. Must be enabled for ``estimators_``
        and ``get_estimator`` to work. Very slow.

    :ivar int n_features_in\_: The number of features (columns) from the fit input ``X``.
    :ivar list feature_names\_: Names for the features of the fit input ``X``.
    :ivar dict ranger_forest\_: The returned result object from calling C++ ranger.
    :ivar int mtry\_: The mtry value as determined if ``mtry`` is callable, otherwise
        it is the same as ``mtry``.
    :ivar float sample_fraction\_: The sample fraction determined by input validation.
    :ivar list regularization_factor\_: The regularization factors determined by input
        validation.
    :ivar list unordered_feature_names\_: The unordered feature names determined by
        input validation.
    :ivar int split_rule\_: The split rule integer corresponding to ranger enum
        ``SplitRule``.
    :ivar bool use_regularization_factor\_: Input validation determined bool for using
        regularization factor input parameter.
    :ivar str respect_categorical_features\_: Input validation determined string
        respecting categorical features.
    :ivar int importance_mode\_: The importance mode integer corresponding to ranger
        enum ``ImportanceMode``.
    :ivar ndarray feature_importances\_: The variable importances from ranger.
    """

    def __init__(
        self,
        n_estimators=100,
        verbose=False,
        mtry=0,
        importance="none",
        min_node_size=0,
        max_depth=0,
        replace=True,
        sample_fraction=None,
        keep_inbag=False,
        inbag=None,
        split_rule="logrank",
        num_random_splits=1,
        alpha=0.5,
        minprop=0.1,
        respect_categorical_features=None,
        scale_permutation_importance=False,
        local_importance=False,
        regularization_factor=None,
        regularization_usedepth=False,
        holdout=False,
        oob_error=False,
        n_jobs=0,
        seed=42,
        enable_tree_details=False,
    ):
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.mtry = mtry
        self.importance = importance
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.replace = replace
        self.sample_fraction = sample_fraction
        self.keep_inbag = keep_inbag
        self.inbag = inbag
        self.split_rule = split_rule
        self.num_random_splits = num_random_splits
        self.alpha = alpha
        self.minprop = minprop
        self.respect_categorical_features = respect_categorical_features
        self.scale_permutation_importance = scale_permutation_importance
        self.local_importance = local_importance
        self.regularization_factor = regularization_factor
        self.regularization_usedepth = regularization_usedepth
        self.holdout = holdout
        self.oob_error = oob_error
        self.n_jobs = n_jobs
        self.seed = seed
        self.enable_tree_details = enable_tree_details

    @property
    def estimators_(self):
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute 'estimators_'"
            ) from None
        if not self.enable_tree_details:
            raise ValueError("enable_tree_details must be True prior to training")
        return [
            RangerTreeSurvival.from_forest(self, idx=idx)
            for idx in range(self.n_estimators)
        ]

    def get_estimator(self, idx):
        """Extract a single estimator tree from the forest.
        :param int idx: The index of the tree to extract.
        """
        check_is_fitted(self)
        if not self.enable_tree_details:
            raise ValueError("enable_tree_details must be True prior to training")
        return RangerTreeSurvival.from_forest(self, idx=idx)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        split_select_weights=None,
        always_split_features=None,
        categorical_features=None,
    ):
        """Fit the ranger random forest using training data.

        :param array2d X: training input features
        :param array2d y: training input targets, rows of (bool, float)
            representing (survival, time)
        :param array1d sample_weight: optional weights for input samples
        :param list split_select_weights: Vector of weights between 0 and 1 of
            probabilities to select features for splitting. Can be a single vector or a
            vector of vectors with one vector per tree.
        :param list always_split_features:  Features which should always be selected for
            splitting. A list of column index values.
        :param list categorical_features: A list of column index values which should be
            considered categorical, or unordered.
        """
        self.tree_type_ = 5  # tree_type, TREE_SURVIVAL

        # Check input
        X = check_array(X)

        # convert 1d array of 2tuples to 2d array
        # ranger expects the time first, and status second
        # since we follow the scikit-survival convention, we fliplr
        y = np.fliplr(np.array(y.tolist()))

        # Check the init parameters
        self._validate_parameters(X, y, sample_weight)

        # Set X info
        self.feature_names_ = [str(c).encode() for c in range(X.shape[1])]
        self._check_n_features(X, reset=True)

        # Check weights
        sample_weight, use_sample_weight = self._check_sample_weight(sample_weight, X)
        (
            always_split_features,
            use_always_split_features,
        ) = self._check_always_split_features(always_split_features)

        (
            categorical_features,
            use_categorical_features,
        ) = self._check_categorical_features(categorical_features)

        (
            split_select_weights,
            use_split_select_weights,
        ) = self._check_split_select_weights(split_select_weights)

        # Fit the forest
        self.ranger_forest_ = ranger.ranger(
            self.tree_type_,
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray(y.astype("float64")),
            self.feature_names_,  # variable_names
            self.mtry_,
            self.n_estimators,  # num_trees
            self.verbose,
            self.seed,
            self.n_jobs_,  # num_threads
            True,  # write_forest
            self.importance_mode_,
            self.min_node_size,
            split_select_weights,
            use_split_select_weights,
            always_split_features,  # always_split_variable_names
            use_always_split_features,  # use_always_split_variable_names
            False,  # prediction_mode
            {},  # loaded_forest
            self.replace,  # sample_with_replacement
            False,  # probability
            categorical_features,  # unordered_feature_names
            use_categorical_features,  # use_unordered_features
            False,  # save_memory
            self.split_rule_,
            sample_weight,  # case_weights
            use_sample_weight,  # use_case_weights
            {},  # class_weights
            False,  # predict_all
            self.keep_inbag,
            self.sample_fraction_,
            self.alpha,
            self.minprop,
            self.holdout,
            1,  # prediction_type
            self.num_random_splits,
            self.oob_error,
            self.max_depth,
            self.inbag or [],
            bool(self.inbag),  # use_inbag
            self.regularization_factor_,
            False,  # use_regularization_factor
            self.regularization_usedepth,
        )
        self.event_times_ = np.array(
            self.ranger_forest_["forest"]["unique_death_times"]
        )
        # dtype to suppress warning about ragged nested sequences
        self.cumulative_hazard_function_ = np.array(
            self.ranger_forest_["forest"]["cumulative_hazard_function"], dtype=object
        )

        if self.enable_tree_details:
            sample_weight = sample_weight if len(sample_weight) > 0 else np.ones(len(X))
            terminal_node_forest = self._get_terminal_node_forest(X)
            terminal_nodes = np.atleast_2d(terminal_node_forest["predictions"]).astype(
                int
            )
            self._set_leaf_samples(terminal_nodes)
            self._set_node_values(y, sample_weight)
            self._set_n_classes()
        return self

    def _predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        result = ranger.ranger(
            self.tree_type_,
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray([[]]),
            self.feature_names_,  # variable_names
            self.mtry_,
            self.n_estimators,  # num_trees
            self.verbose,
            self.seed,
            self.n_jobs_,  # num_threads
            False,  # write_forest
            self.importance_mode_,
            self.min_node_size,
            [],
            False,  # use_split_select_weights
            [],  # always_split_variable_names
            False,  # use_always_split_variable_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            self.replace,  # sample_with_replacement
            False,  # probability
            [],  # unordered_feature_names
            False,  # use_unordered_features
            False,  # save_memory
            self.split_rule_,
            [],  # case_weights
            False,  # use_case_weights
            {},  # class_weights
            False,  # predict_all
            self.keep_inbag,
            [1],  # sample_fraction
            self.alpha,
            self.minprop,
            self.holdout,
            1,  # prediction_type
            self.num_random_splits,
            self.oob_error,
            self.max_depth,
            self.inbag or [],
            bool(self.inbag),  # use_inbag
            self.regularization_factor_,
            self.use_regularization_factor_,
            self.regularization_usedepth,
        )
        return result

    def predict_cumulative_hazard_function(self, X):
        """Predict cumulative hazard function.

        :param array2d X: prediction input features
        """
        result = self._predict(X)
        return np.atleast_2d(result["predictions"])

    def predict_survival_function(self, X):
        """Predict survival function.

        :param array2d X: prediction input features
        """
        chf = self.predict_cumulative_hazard_function(X)
        return np.exp(-chf)

    def predict(self, X):
        """Predict risk score.

        :param array2d X: prediction input features
        """
        chf = self.predict_cumulative_hazard_function(X)
        return chf.sum(1)

    def _more_tags(self):
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            },
        }
