"""Scikit-learn wrapper for ranger classification."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from skranger import ranger
from skranger.ensemble.base import BaseRangerForest
from skranger.tree.classifier import RangerTreeClassifier


class RangerForestClassifier(BaseRangerForest, ClassifierMixin, BaseEstimator):
    r"""Ranger Random Forest Probability/Classification implementation for sci-kit learn.

    Provides a sklearn classifier interface to the Ranger C++ library using Cython.

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
    :param float/list sample_fraction: The fraction of observations to sample. The
        default is 1 when sampling with replacement, and 0.632 otherwise. This can be a
        list of class specific values.
    :param bool keep_inbag: If true, save how often observations are in-bag in each
        tree. These will be stored in the ``ranger_forest_`` attribute under the key
        ``"inbag_counts"``.
    :param list inbag: A list of size ``n_estimators``, containing inbag counts for each
        observation. Can be used for stratified sampling.
    :param str split_rule: One of ``gini``, ``extratrees``, ``hellinger``;
        default ``gini``.
    :param int num_random_splits: The number of random splits to consider for the
        ``extratrees`` splitrule.
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
    :param bool save_memory: Save memory at the cost of speed growing trees.
    :param int seed: Random seed value.
    :param bool enable_tree_details: When ``True``, perform additional calculations
        for detailing the underlying decision trees. Must be enabled for ``estimators_``
        and ``get_estimator`` to work. Very slow.

    :ivar ndarray classes\_: The class labels determined from the fit input ``y``.
    :ivar int n_classes\_: The number of unique class labels from the fit input ``y``.
    :ivar int n_features_in\_: The number of features (columns) from the fit input
        ``X``.
    :ivar list feature_names\_: Names for the features of the fit input ``X``.
    :ivar dict ranger_forest\_: The returned result object from calling C++ ranger.
    :ivar int mtry\_: The mtry value as determined if ``mtry`` is callable, otherwise
        it is the same as ``mtry``.
    :ivar float/list sample_fraction\_: The sample fraction determined by input
        validation
    :ivar list regularization_factor\_: The regularization factors determined by input
        validation.
    :ivar int split_rule\_: The split rule integer corresponding to ranger enum
        ``SplitRule``.
    :ivar bool use_regularization_factor\_: Input validation determined bool for using
        regularization factor input parameter.
    :ivar str respect_categorical_features\_: Input validation determined string
        respecting categorical features.
    :ivar int importance_mode\_: The importance mode integer corresponding to ranger
        enum ``ImportanceMode``.
    :ivar list ranger_class_order\_: The class reference ordering derived from ranger.
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
        split_rule="gini",
        num_random_splits=1,
        respect_categorical_features=None,
        scale_permutation_importance=False,
        local_importance=False,
        regularization_factor=None,
        regularization_usedepth=False,
        holdout=False,
        oob_error=False,
        n_jobs=-1,
        save_memory=False,
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
        self.respect_categorical_features = respect_categorical_features
        self.scale_permutation_importance = scale_permutation_importance
        self.local_importance = local_importance
        self.regularization_factor = regularization_factor
        self.regularization_usedepth = regularization_usedepth
        self.holdout = holdout
        self.oob_error = oob_error
        self.n_jobs = n_jobs
        self.save_memory = save_memory
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
            RangerTreeClassifier.from_forest(self, idx=idx)
            for idx in range(self.n_estimators)
        ]

    def get_estimator(self, idx):
        """Extract a single estimator tree from the forest.
        :param int idx: The index of the tree to extract.
        """
        check_is_fitted(self)
        if not self.enable_tree_details:
            raise ValueError("enable_tree_details must be True prior to training")
        return RangerTreeClassifier.from_forest(self, idx=idx)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        class_weights=None,
        split_select_weights=None,
        always_split_features=None,
        categorical_features=None,
    ):
        """Fit the ranger random forest using training data.

        :param array2d X: training input features
        :param array1d y: training input target classes
        :param array1d sample_weight: optional weights for input samples
        :param dict class_weights: A dictionary of outcome classes to weights.
        :param list split_select_weights: Vector of weights between 0 and 1 of
            probabilities to select features for splitting. Can be a single vector or a
            vector of vectors with one vector per tree.
        :param list always_split_features:  Features which should always be selected for
            splitting. A list of column index values.
        :param list categorical_features: A list of column index values which should be
            considered categorical, or unordered.
       """
        self.tree_type_ = 9  # tree_type, TREE_PROBABILITY enables predict_proba

        # Check input
        X, y = self._validate_data(X, y)
        check_classification_targets(y)

        # Check the init parameters
        self._validate_parameters(X, y, sample_weight)

        # Map classes to indices
        y = np.copy(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        if class_weights is None:
            class_weights = {}
        else:
            try:
                class_weights = {
                    idx: class_weights[k] for idx, k in enumerate(self.classes_)
                }
            except KeyError:
                raise ValueError(
                    "class weights must have a weight for each class"
                ) from None

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
            split_select_weights,
            use_split_select_weights,
            always_split_features,  # always_split_variable_names
            bool(always_split_features),  # use_always_split_variable_names
            False,  # prediction_mode
            {},  # loaded_forest
            self.replace,  # sample_with_replacement
            False,  # probability
            categorical_features,  # unordered_variable_names
            use_categorical_features,  # use_unordered_variable_names
            self.save_memory,
            self.split_rule_,
            sample_weight,  # case_weights
            use_sample_weight,  # use_case_weights
            class_weights,
            False,  # predict_all
            self.keep_inbag,
            self.sample_fraction_,
            0.5,  # alpha, ignored because maxstat can't be used on classification
            0.1,  # minprop, ignored because maxstat can't be used on classification
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
        self.ranger_class_order_ = np.argsort(
            np.array(self.ranger_forest_["forest"]["class_values"]).astype(int)
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

    def predict(self, X):
        """Predict classes from X.

        :param array2d X: prediction input features
        """
        probas = self.predict_proba(X)
        return self.classes_.take(np.argmax(probas, axis=1), axis=0)

    def predict_proba(self, X):
        """Predict probabilities for classes from X.

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
            [],  # split_select_weights
            False,  # use_split_select_weights
            [],  # always_split_variable_names
            False,  # use_always_split_variable_names
            True,  # prediction_mode
            self.ranger_forest_["forest"],  # loaded_forest
            self.replace,  # sample_with_replacement
            False,  # probability
            [],  # unordered_feature_names
            False,  # use_unordered_features
            self.save_memory,
            self.split_rule_,
            [],  # case_weights
            False,  # use_case_weights
            {},  # class_weights
            False,  # predict_all
            self.keep_inbag,
            [1],  # sample_fraction
            0.5,  # alpha
            0.1,  # minprop
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
        predictions = np.atleast_2d(np.array(result["predictions"]))
        return predictions[:, self.ranger_class_order_]

    def predict_log_proba(self, X):
        """Predict log probabilities for classes from X.

        :param array2d X: prediction input features
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            }
        }
