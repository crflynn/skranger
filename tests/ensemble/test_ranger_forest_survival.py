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
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        with pytest.raises(NotFittedError):
            check_is_fitted(rfs)
        rfs.fit(lung_X, lung_y)
        check_is_fitted(rfs)
        assert hasattr(rfs, "event_times_")
        assert hasattr(rfs, "cumulative_hazard_function_")
        assert hasattr(rfs, "ranger_forest_")
        assert hasattr(rfs, "n_features_in_")

    def test_predict(self, lung_X, lung_y):
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        rfs.fit(lung_X, lung_y)
        pred = rfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

        # test with single record
        lung_X_record = lung_X.values[0:1, :]
        pred = rfs.predict(lung_X_record)
        assert len(pred) == 1

    def test_predict_cumulative_hazard_function(self, lung_X, lung_y):
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        rfs.fit(lung_X, lung_y)
        pred = rfs.predict_cumulative_hazard_function(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_predict_survival_function(self, lung_X, lung_y):
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        rfs.fit(lung_X, lung_y)
        pred = rfs.predict_survival_function(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        tf = tempfile.TemporaryFile()
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        rfs.fit(lung_X, lung_y)
        pickle.dump(rfs, tf)
        tf.seek(0)
        new_rfs = pickle.load(tf)
        pred = new_rfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        rfs.fit(lung_X, lung_y)
        clone(rfs)

    def test_verbose(self, lung_X, lung_y, verbose, capfd):
        rfs = RangerForestSurvival(verbose=verbose)
        rfs.fit(lung_X, lung_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(self, lung_X, lung_y, importance, scale_permutation_importance, local_importance):
        rfs = RangerForestSurvival(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)
            return

        rfs.fit(lung_X, lung_y)
        if importance == "none":
            assert rfs.importance_mode_ == 0
        elif importance == "impurity":
            assert rfs.importance_mode_ == 1
        elif importance == "impurity_corrected":
            assert rfs.importance_mode_ == 5
        elif importance == "permutation":
            if local_importance:
                assert rfs.importance_mode_ == 6
            elif scale_permutation_importance:
                assert rfs.importance_mode_ == 2
            else:
                assert rfs.importance_mode_ == 3

    def test_mtry(self, lung_X, lung_y, mtry):
        rfs = RangerForestSurvival(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > lung_X.shape[0]):
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)
            return

        rfs.fit(lung_X, lung_y)
        if callable(mtry):
            assert rfs.mtry_ == mtry(lung_X.shape[1])
        else:
            assert rfs.mtry_ == mtry

    def test_inbag(self, lung_X, lung_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        rfs = RangerForestSurvival(n_estimators=2, inbag=inbag)
        rfs.fit(lung_X, lung_y)

        # inbag list different length from n_estimators
        rfs = RangerForestSurvival(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            rfs.fit(lung_X, lung_y)

        # can't use inbag with sample weight
        rfs = RangerForestSurvival(inbag=inbag)
        with pytest.raises(ValueError):
            rfs.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))

        # can't use class sampling and inbag
        rfs = RangerForestSurvival(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            rfs.fit(lung_X, lung_y)

    def test_sample_fraction(self, lung_X, lung_y):
        rfs = RangerForestSurvival(sample_fraction=0.69)
        rfs.fit(lung_X, lung_y)
        assert rfs.sample_fraction_ == [0.69]

        # test with single record
        lung_X_record = lung_X.values[0:1, :]
        pred = rfs.predict(lung_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, lung_X, lung_y, replace):
        rfs = RangerForestSurvival(replace=replace)
        rfs.fit(lung_X, lung_y)

        if replace:
            assert rfs.sample_fraction_ == [1.0]
        else:
            assert rfs.sample_fraction_ == [0.632]

    def test_categorical_features(self, lung_X, lung_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(np.array([random.choice([0, 1]) for _ in range(lung_X.shape[0])]))
        lung_X_c = np.hstack((lung_X, categorical_col.transpose()))
        categorical_features = [lung_X.shape[1]]

        rfs = RangerForestSurvival(
            respect_categorical_features=respect_categorical_features, categorical_features=categorical_features
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                rfs.fit(lung_X_c, lung_y)
            return

        rfs.fit(lung_X_c, lung_y)

        if respect_categorical_features in ("ignore", "order"):
            assert rfs.categorical_features_ == []
        else:
            assert rfs.categorical_features_ == [str(c).encode() for c in categorical_features]

    def test_split_rule(self, lung_X, lung_y, split_rule):
        rfs = RangerForestSurvival(split_rule=split_rule)

        if split_rule not in ["logrank", "extratrees", "C", "C_ignore_ties", "maxstat"]:
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)
            return

        rfs.fit(lung_X, lung_y)

        if split_rule == "logrank":
            assert rfs.split_rule_ == 1
        elif split_rule == "extratrees":
            assert rfs.split_rule_ == 5
        elif split_rule == "C":
            assert rfs.split_rule_ == 2
        elif split_rule == "C_ignore_ties":
            assert rfs.split_rule_ == 3
        elif split_rule == "maxstat":
            assert rfs.split_rule_ == 4

        if split_rule != "extratrees":
            rfs = RangerForestSurvival(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)

    def test_regularization(self, lung_X, lung_y):
        rfs = RangerForestSurvival()
        rfs.fit(lung_X, lung_y)
        assert rfs.regularization_factor_ == []
        assert not rfs.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            rfs = RangerForestSurvival(regularization_factor=r)
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)

        # vector of ones isn't applied
        rfs = RangerForestSurvival(regularization_factor=[1] * lung_X.shape[1])
        rfs.fit(lung_X, lung_y)
        assert rfs.regularization_factor_ == []
        assert not rfs.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        rfs = RangerForestSurvival(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            rfs.fit(lung_X, lung_y)
        assert rfs.n_jobs_ == 1
        assert rfs.regularization_factor_ == reg
        assert rfs.use_regularization_factor_

    def test_always_split_features(self, lung_X, lung_y):
        rfs = RangerForestSurvival(always_split_features=[0])
        rfs.fit(lung_X, lung_y)
        # feature 0 is in every tree split
        for tree in rfs.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_feature_importances_(self, lung_X, lung_y, importance, local_importance):
        rfs = RangerForestSurvival(importance=importance, local_importance=local_importance)
        with pytest.raises(AttributeError):
            _ = rfs.feature_importances_

        if importance == "INVALID":
            with pytest.raises(ValueError):
                rfs.fit(lung_X, lung_y)
            return

        rfs.fit(lung_X, lung_y)
        if importance == "none":
            with pytest.raises(ValueError):
                _ = rfs.feature_importances_

    def test_sample_weight(self, lung_X, lung_y):
        rfs_w = RangerForestSurvival()
        rfs_w.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))
        rfs = RangerForestSurvival()
        rfs.fit(lung_X, lung_y)

        pred_w = rfs_w.predict(lung_X)
        pred = rfs.predict(lung_X)

        np.testing.assert_array_equal(pred.reshape(-1, 1), pred_w.reshape(-1, 1))

    def test_get_tags(self):
        rfs = RangerForestSurvival()
        tags = rfs._get_tags()
        assert tags["requires_y"]

    # We can't check this because we conform to scikit-survival api,
    # rather than scikit-learn's
    # def test_check_estimator(self):
    #     check_estimator(RangerForestSurvival())
