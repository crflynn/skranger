import pickle
import random
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skranger.ensemble import RangerForestSurvival

N_ESTIMATORS = 10


class TestRangerForestSurvival:
    def test_init(self):
        _ = RangerForestSurvival()

    def test_fit(self, lung_X, lung_y):
        forest = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(lung_X, lung_y)
        check_is_fitted(forest)
        assert hasattr(forest, "event_times_")
        assert hasattr(forest, "cumulative_hazard_function_")
        assert hasattr(forest, "ranger_forest_")
        assert hasattr(forest, "n_features_in_")

    def test_predict(self, lung_X, lung_y):
        forest = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        forest.fit(lung_X, lung_y)
        pred = forest.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

        # test with single record
        lung_X_record = lung_X.values[0:1, :]
        pred = forest.predict(lung_X_record)
        assert len(pred) == 1

    def test_predict_cumulative_hazard_function(self, lung_X, lung_y):
        forest = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        forest.fit(lung_X, lung_y)
        pred = forest.predict_cumulative_hazard_function(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_predict_survival_function(self, lung_X, lung_y):
        forest = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        forest.fit(lung_X, lung_y)
        pred = forest.predict_survival_function(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        tf = tempfile.TemporaryFile()
        forest = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        forest.fit(lung_X, lung_y)
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        forest = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        forest.fit(lung_X, lung_y)
        clone(forest)

    def test_verbose(self, lung_X, lung_y, verbose, capfd):
        forest = RangerForestSurvival(verbose=verbose)
        forest.fit(lung_X, lung_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(
        self, lung_X, lung_y, importance, scale_permutation_importance, local_importance
    ):
        forest = RangerForestSurvival(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
            return

        forest.fit(lung_X, lung_y)
        if importance == "none":
            assert forest.importance_mode_ == 0
        elif importance == "impurity":
            assert forest.importance_mode_ == 1
        elif importance == "impurity_corrected":
            assert forest.importance_mode_ == 5
        elif importance == "permutation":
            if local_importance:
                assert forest.importance_mode_ == 6
            elif scale_permutation_importance:
                assert forest.importance_mode_ == 2
            else:
                assert forest.importance_mode_ == 3

    def test_importance_pvalues(self, lung_X_mod, lung_y, importance, mod):
        forest = RangerForestSurvival(importance=importance)
        np.random.seed(42)

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                forest.fit(lung_X_mod, lung_y)
            return

        if not importance == "impurity_corrected":
            forest.fit(lung_X_mod, lung_y)
            with pytest.raises(ValueError):
                forest.get_importance_pvalues()
            return

        # Test error for no non-negative importance values
        if mod == "none":
            forest.fit(lung_X_mod, lung_y)
            with pytest.raises(ValueError):
                forest.get_importance_pvalues()
            return

        forest.fit(lung_X_mod, lung_y)
        assert len(forest.get_importance_pvalues()) == lung_X_mod.shape[1]

    def test_mtry(self, lung_X, lung_y, mtry):
        forest = RangerForestSurvival(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > lung_X.shape[0]):
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
            return

        forest.fit(lung_X, lung_y)
        if callable(mtry):
            assert forest.mtry_ == mtry(lung_X.shape[1])
        else:
            assert forest.mtry_ == mtry

    def test_inbag(self, lung_X, lung_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        forest = RangerForestSurvival(n_estimators=2, inbag=inbag)
        forest.fit(lung_X, lung_y)

        # inbag list different length from n_estimators
        forest = RangerForestSurvival(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            forest.fit(lung_X, lung_y)

        # can't use inbag with sample weight
        forest = RangerForestSurvival(inbag=inbag)
        with pytest.raises(ValueError):
            forest.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))

        # can't use class sampling and inbag
        forest = RangerForestSurvival(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            forest.fit(lung_X, lung_y)

    def test_sample_fraction(self, lung_X, lung_y):
        forest = RangerForestSurvival(sample_fraction=0.69)
        forest.fit(lung_X, lung_y)
        assert forest.sample_fraction_ == [0.69]

        # test with single record
        lung_X_record = lung_X.values[0:1, :]
        pred = forest.predict(lung_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, lung_X, lung_y, replace):
        forest = RangerForestSurvival(replace=replace)
        forest.fit(lung_X, lung_y)

        if replace:
            assert forest.sample_fraction_ == [1.0]
        else:
            assert forest.sample_fraction_ == [0.632]

    def test_categorical_features(self, lung_X, lung_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(lung_X.shape[0])])
        )
        lung_X_c = np.hstack((lung_X, categorical_col.transpose()))
        categorical_features = [lung_X.shape[1]]

        forest = RangerForestSurvival(
            respect_categorical_features=respect_categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                forest.fit(lung_X_c, lung_y, categorical_features=categorical_features)
            return

        forest.fit(lung_X_c, lung_y, categorical_features=categorical_features)
        forest.predict(lung_X_c)

    def test_split_rule(self, lung_X, lung_y, split_rule):
        forest = RangerForestSurvival(split_rule=split_rule)

        if split_rule not in ["logrank", "extratrees", "C", "C_ignore_ties", "maxstat"]:
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
            return

        forest.fit(lung_X, lung_y)

        if split_rule == "logrank":
            assert forest.split_rule_ == 1
        elif split_rule == "extratrees":
            assert forest.split_rule_ == 5
        elif split_rule == "C":
            assert forest.split_rule_ == 2
        elif split_rule == "C_ignore_ties":
            assert forest.split_rule_ == 3
        elif split_rule == "maxstat":
            assert forest.split_rule_ == 4

        if split_rule != "extratrees":
            forest = RangerForestSurvival(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)

    def test_split_select_weights(self, lung_X, lung_y):
        n_trees = 10
        weights = [0.1] * lung_X.shape[1]
        forest = RangerForestSurvival(n_estimators=n_trees,)
        forest.fit(lung_X, lung_y, split_select_weights=weights)

        weights = [0.1] * (lung_X.shape[1] - 1)
        forest = RangerForestSurvival(n_estimators=n_trees)

        with pytest.raises(RuntimeError):
            forest.fit(lung_X, lung_y, split_select_weights=weights)

        weights = [[0.1] * (lung_X.shape[1])] * n_trees
        forest = RangerForestSurvival(n_estimators=n_trees)
        forest.fit(lung_X, lung_y, split_select_weights=weights)

        weights = [[0.1] * (lung_X.shape[1])] * (n_trees + 1)
        forest = RangerForestSurvival(n_estimators=n_trees)
        with pytest.raises(RuntimeError):
            forest.fit(lung_X, lung_y, split_select_weights=weights)

    def test_regularization(self, lung_X, lung_y):
        forest = RangerForestSurvival()
        forest.fit(lung_X, lung_y)
        assert forest.regularization_factor_ == []
        assert not forest.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            forest = RangerForestSurvival(regularization_factor=r)
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)

        # vector of ones isn't applied
        forest = RangerForestSurvival(regularization_factor=[1] * lung_X.shape[1])
        forest.fit(lung_X, lung_y)
        assert forest.regularization_factor_ == []
        assert not forest.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        forest = RangerForestSurvival(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            forest.fit(lung_X, lung_y)
        assert forest.n_jobs_ == 1
        assert forest.regularization_factor_ == reg
        assert forest.use_regularization_factor_

    def test_always_split_features(self, lung_X, lung_y):
        forest = RangerForestSurvival()
        forest.fit(lung_X, lung_y, always_split_features=[0])
        # feature 0 is in every tree split
        for tree in forest.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_feature_importances_(self, lung_X, lung_y, importance, local_importance):
        forest = RangerForestSurvival(
            importance=importance, local_importance=local_importance
        )
        with pytest.raises(AttributeError):
            _ = forest.feature_importances_

        if importance == "INVALID":
            with pytest.raises(ValueError):
                forest.fit(lung_X, lung_y)
            return

        forest.fit(lung_X, lung_y)
        if importance == "none":
            with pytest.raises(ValueError):
                _ = forest.feature_importances_
        else:
            assert len(forest.feature_importances_) == lung_X.shape[1]

    def test_sample_weight(self, lung_X, lung_y):
        forest_w = RangerForestSurvival()
        forest_w.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))
        forest = RangerForestSurvival()
        forest.fit(lung_X, lung_y)

        pred_w = forest_w.predict(lung_X)
        pred = forest.predict(lung_X)

        np.testing.assert_array_equal(pred.reshape(-1, 1), pred_w.reshape(-1, 1))

    def test_get_tags(self):
        forest = RangerForestSurvival()
        tags = forest._get_tags()
        assert tags["requires_y"]

    # We can't check this because we conform to scikit-survival api,
    # rather than scikit-learn's
    # def test_check_estimator(self):
    #     check_estimator(RangerForestSurvival())
