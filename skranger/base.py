import warnings
from typing import Iterable

import numpy as np
from sklearn.utils.validation import _check_sample_weight


class RangerMixin:
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
        self.order_snps_ = self.respect_categorical_features_ == "order"

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
