"""Scikit-learn wrapper for ranger regression."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skranger.ensemble import ranger
from skranger.ensemble.base import RangerMixin


class RangerForestRegressor(RangerMixin, RegressorMixin, BaseEstimator):
    r"""Ranger Random Forest Regression implementation for sci-kit learn.

    Provides a sklearn regressor interface to the Ranger C++ library using Cython. The
    argument names to the constructor are similar to the C++ library and accompanied R
    package for familiarity.

    :param int n_estimators: The number of tree regressors to train
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
    :param float/list sample_fraction: The fraction of observations to sample. The
        default is 1 when sampling with replacement, and 0.632 otherwise. This can be a
        list of class specific values.
    :param bool keep_inbag: If true, save how often observations are in-bag in each
        tree. These will be stored in the ``ranger_forest_`` attribute under the key
        ``"inbag_counts"``.
    :param list inbag: A list of size ``n_estimators``, containing inbag counts for each
        observation. Can be used for stratified sampling.
    :param str split_rule: One of ``variance``, ``extratrees``, ``maxstat``, ``beta``;
        default ``variance``.
    :param int num_random_splits: The number of trees for the ``extratrees`` splitrule.
    :param float alpha: Significance threshold to allow splitting for the ``maxstat``
        split rule.
    :param float minprop: Lower quantile of covariate distribution to be considered for
        splitting for ``maxstat`` split rule.
    :param list split_select_weights: Vector of weights between 0 and 1 of probabilities
        to select features for splitting.
    :param list always_split_features:  Features which should always be selected for
        splitting. A list of column index values.
    :param list categorical_features: A list of column index values which should be
        considered categorical, or unordered.
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
    :param bool quantiles: Enable quantile regression after fitting. This must be
        set to ``True`` in order to call ``predict_quantiles`` after fitting.
    :param bool oob_error: Whether to calculate out-of-bag prediction error.
    :param int n_jobs: The number of threads. Default is number of CPU cores.
    :param bool save_memory: Save memory at the cost of speed growing trees.
    :param int seed: Random seed value.

    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
    :ivar list feature_names\_: Names for the features of the fit input ``X``.
    :ivar dict ranger_forest\_: The returned result object from calling C++ ranger.
    :ivar int mtry\_: The mtry value as determined if ``mtry`` is callable, otherwise
        it is the same as ``mtry``.
    :ivar float sample_fraction\_: The sample fraction determined by input validation
    :ivar list regularization_factor\_: The regularization factors determined by input
        validation.
    :ivar list unordered_features\_: The unordered feature names determined by
        input validation.
    :ivar int split_rule\_: The split rule integer corresponding to ranger enum
        ``SplitRule``.
    :ivar bool use_regularization_factor\_: Input validation determined bool for using
        regularization factor input parameter.
    :ivar str respect_categorical_features\_: Input validation determined string
        respecting categorical features.
    :ivar int importance_mode\_: The importance mode integer corresponding to ranger
        enum ``ImportanceMode``.
    :ivar 2darray random_node_values\_: Random training target values based on
        trained forest terminal nodes for the purpose of quantile regression.
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
        split_rule="variance",
        num_random_splits=1,
        alpha=0.5,
        minprop=0.1,
        split_select_weights=None,
        always_split_features=None,
        categorical_features=None,
        respect_categorical_features=None,
        scale_permutation_importance=False,
        local_importance=False,
        regularization_factor=None,
        regularization_usedepth=False,
        holdout=False,
        quantiles=False,
        oob_error=False,
        n_jobs=-1,
        save_memory=False,
        seed=42,
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
        self.split_select_weights = split_select_weights
        self.always_split_features = always_split_features
        self.categorical_features = categorical_features
        self.respect_categorical_features = respect_categorical_features
        self.scale_permutation_importance = scale_permutation_importance
        self.local_importance = local_importance
        self.regularization_factor = regularization_factor
        self.regularization_usedepth = regularization_usedepth
        self.holdout = holdout
        self.quantiles = quantiles
        self.oob_error = oob_error
        self.n_jobs = n_jobs
        self.save_memory = save_memory
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        """Fit the ranger random forest using training data.

        :param array2d X: training input features
        :param array1d y: training input targets
        :param array1d sample_weight: optional weights for input samples
        """
        self.tree_type_ = 3  # tree_type, TREE_REGRESSION

        # Check input
        X, y = self._validate_data(X, y)

        # Check the init parameters
        self._validate_parameters(X, y, sample_weight)

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

        # Set X info
        self.feature_names_ = [str(c).encode() for c in range(X.shape[1])]
        self._check_n_features(X, reset=True)

        if self.always_split_features is not None:
            always_split_features = [
                str(c).encode() for c in self.always_split_features
            ]
        else:
            always_split_features = []

        # Fit the forest
        self.ranger_forest_ = ranger.ranger(
            self.tree_type_,
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray(np.atleast_2d(y).astype("float64").transpose()),
            self.feature_names_,  # variable_names
            self.mtry_,
            self.n_estimators,  # num_trees
            self.verbose,
            self.seed,
            self.n_jobs_,  # num_threads
            True,  # write_forest
            self.importance_mode_,
            self.min_node_size,
            self.split_select_weights or [],
            bool(self.split_select_weights),  # use_split_select_weights
            always_split_features,  # always_split_feature_names
            bool(always_split_features),  # use_always_split_feature_names
            False,  # prediction_mode
            {},  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            self.replace,  # sample_with_replacement
            False,  # probability
            self.categorical_features_,  # unordered_feature_names
            bool(self.categorical_features_),  # use_unordered_features
            self.save_memory,
            self.split_rule_,
            sample_weight,  # case_weights
            use_sample_weight,  # use_case_weights
            [],  # class_weights
            False,  # predict_all
            self.keep_inbag,
            self.sample_fraction_,
            self.alpha,
            self.minprop,
            self.holdout,
            1,  # prediction_type
            self.num_random_splits,
            False,  # use_sparse_data
            self.order_snps_,
            self.oob_error,
            self.max_depth,
            self.inbag or [],
            bool(self.inbag),  # use_inbag
            self.regularization_factor_,
            False,  # use_regularization_factor
            self.regularization_usedepth,
        )

        if self.quantiles:
            forest = self._get_terminal_node_forest(X)
            terminal_nodes = np.array(forest["predictions"]).astype(int)
            self.random_node_values_ = np.empty(
                (np.max(terminal_nodes) + 1, self.n_estimators)
            )
            self.random_node_values_[:] = np.nan
            for tree in range(self.n_estimators):
                idx = np.arange(X.shape[0])
                np.random.shuffle(idx)
                self.random_node_values_[terminal_nodes[idx, tree], tree] = y[idx]

        return self

    def _get_terminal_node_forest(self, X):
        """Get a terminal node forest for X.

        :param array2d X: prediction input features
        """
        # many fields defaulted here which are unused
        forest = ranger.ranger(
            self.tree_type_,
            np.asfortranarray(X.astype("float64")),
            np.asfortranarray([[]]),
            self.feature_names_,  # variable_names
            0,  # m_try
            self.n_estimators,  # num_trees
            self.verbose,
            self.seed,
            self.n_jobs_,  # num_threads
            False,  # write_forest
            0,  # importance_mode
            0,  # min_node_size
            [],  # split_select_weights
            False,  # use_split_select_weights
            [],  # always_split_feature_names
            False,  # use_always_split_feature_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            True,  # sample_with_replacement
            False,  # probability
            [],  # unordered_feature_names
            False,  # use_unordered_features
            False,  # save_memory
            1,  # split_rule
            [],  # case_weights
            False,  # use_case_weights
            [],  # class_weights
            False,  # predict_all
            self.keep_inbag,
            [1],  # sample_fraction
            0,  # alpha
            0,  # minprop
            self.holdout,
            2,  # prediction_type (terminal nodes)
            1,  # num_random_splits
            False,  # use_sparse_data
            False,  # order_snps_
            False,  # oob_error
            0,  # max_depth
            [],  # inbag
            False,  # use_inbag
            [],  # regularization_factor_
            False,  # use_regularization_factor_
            False,  # regularization_usedepth
        )
        return forest

    def predict_quantiles(self, X, quantiles=None):
        """Predict quantile regression target for X.

        :param array2d X: prediction input features
        :param list(float) quantiles: a list of quantiles on which to predict.
          If the list contains a single quantile, the result will be a 1darray.
          If there are multiple quantiles, the result will be a 2darray with
          columns corresponding to respective quantiles. Default is ``[0.1, 0.5, 0.9]``.
        """
        if not hasattr(self, "random_node_values_"):
            raise ValueError("Must set quantiles = True for quantile predictions.")
        quantiles = quantiles or [0.1, 0.5, 0.9]
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        forest = self._get_terminal_node_forest(X)
        terminal_nodes = np.array(forest["predictions"]).astype(int)
        node_values = 0.0 * terminal_nodes
        for tree in range(self.n_estimators):
            node_values[:, tree] = self.random_node_values_[
                terminal_nodes[:, tree], tree
            ]
        quantile_predictions = np.nanquantile(node_values, quantiles, axis=1)
        if len(quantiles) == 1:
            return np.squeeze(quantile_predictions)
        return quantile_predictions

    def predict(self, X):
        """Predict regression target for X.

        :param array2d X: prediction input features
        """
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
            self.split_select_weights or [],
            bool(self.split_select_weights),  # use_split_select_weights
            [],  # always_split_feature_names
            False,  # use_always_split_feature_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            np.asfortranarray([[]]),  # snp_data
            self.replace,  # sample_with_replacement
            False,  # probability
            self.categorical_features_,  # unordered_feature_names
            bool(self.categorical_features_),  # use_unordered_features
            self.save_memory,
            self.split_rule_,
            [],  # case_weights
            False,  # use_case_weights
            [],  # class_weights
            False,  # predict_all
            self.keep_inbag,
            [1],  # sample_fraction
            self.alpha,
            self.minprop,
            self.holdout,
            1,  # prediction_type
            self.num_random_splits,
            False,  # use_sparse_data
            self.order_snps_,
            self.oob_error,
            self.max_depth,
            self.inbag or [],
            bool(self.inbag),  # use_inbag
            self.regularization_factor_,
            self.use_regularization_factor_,
            self.regularization_usedepth,
        )
        return np.array(result["predictions"])

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            }
        }
