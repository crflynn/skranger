import pickle
import random
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
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
        assert hasattr(rfs, "n_features_")

    def test_predict(self, lung_X, lung_y):
        rfs = RangerForestSurvival(n_estimators=N_ESTIMATORS)
        rfs.fit(lung_X, lung_y)
        pred = rfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

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
        rfc = RangerForestSurvival(verbose=verbose)
        rfc.fit(lung_X, lung_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(self, lung_X, lung_y, importance, scale_permutation_importance, local_importance):
        rfc = RangerForestSurvival(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfc.fit(lung_X, lung_y)
            return

        rfc.fit(lung_X, lung_y)
        if importance == "none":
            assert rfc.importance_mode_ == 0
        elif importance == "impurity":
            assert rfc.importance_mode_ == 1
        elif importance == "impurity_corrected":
            assert rfc.importance_mode_ == 5
        elif importance == "permutation":
            if local_importance:
                assert rfc.importance_mode_ == 6
            elif scale_permutation_importance:
                assert rfc.importance_mode_ == 2
            else:
                assert rfc.importance_mode_ == 3

    def test_mtry(self, lung_X, lung_y, mtry):
        rfc = RangerForestSurvival(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                rfc.fit(lung_X, lung_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > lung_X.shape[0]):
            with pytest.raises(ValueError):
                rfc.fit(lung_X, lung_y)
            return

        rfc.fit(lung_X, lung_y)
        if callable(mtry):
            assert rfc.mtry_ == mtry(lung_X.shape[1])
        else:
            assert rfc.mtry_ == mtry

    def test_inbag(self, lung_X, lung_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        rfc = RangerForestSurvival(n_estimators=2, inbag=inbag)
        rfc.fit(lung_X, lung_y)

        # inbag list different length from n_estimators
        rfc = RangerForestSurvival(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            rfc.fit(lung_X, lung_y)

        # can't use inbag with sample weight
        rfc = RangerForestSurvival(inbag=inbag)
        with pytest.raises(ValueError):
            rfc.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))

        # can't use class sampling and inbag
        rfc = RangerForestSurvival(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            rfc.fit(lung_X, lung_y)

    def test_sample_fraction(self, iris_X, iris_y):
        rfs = RangerForestSurvival(sample_fraction=0.69)
        rfs.fit(iris_X, iris_y)
        assert rfs.sample_fraction_ == [0.69]

    def test_sample_fraction_replace(self, lung_X, lung_y, replace):
        rfc = RangerForestSurvival(replace=replace)
        rfc.fit(lung_X, lung_y)

        if replace:
            assert rfc.sample_fraction_ == [1.0]
        else:
            assert rfc.sample_fraction_ == [0.632]

    def test_categorical_features(self, lung_X, lung_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(np.array([random.choice([0, 1]) for _ in range(lung_X.shape[0])]))
        lung_X_c = np.hstack((lung_X, categorical_col.transpose()))
        categorical_features = [lung_X.shape[1]]

        rfc = RangerForestSurvival(
            respect_categorical_features=respect_categorical_features, categorical_features=categorical_features
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                rfc.fit(lung_X_c, lung_y)
            return

        rfc.fit(lung_X_c, lung_y)

        if respect_categorical_features in ("ignore", "order"):
            assert rfc.categorical_features_ == []
        else:
            assert rfc.categorical_features_ == [str(c).encode() for c in categorical_features]

    def test_split_rule(self, lung_X, lung_y, split_rule):
        rfc = RangerForestSurvival(split_rule=split_rule)

        if split_rule not in ["logrank", "extratrees", "C", "C_ignore_ties", "maxstat"]:
            with pytest.raises(ValueError):
                rfc.fit(lung_X, lung_y)
            return

        rfc.fit(lung_X, lung_y)

        if split_rule == "logrank":
            assert rfc.split_rule_ == 1
        elif split_rule == "extratrees":
            assert rfc.split_rule_ == 5
        elif split_rule == "C":
            assert rfc.split_rule_ == 2
        elif split_rule == "C_ignore_ties":
            assert rfc.split_rule_ == 3
        elif split_rule == "maxstat":
            assert rfc.split_rule_ == 4

        if split_rule != "extratrees":
            rfc = RangerForestSurvival(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                rfc.fit(lung_X, lung_y)

    def test_regularization(self, lung_X, lung_y):
        rfc = RangerForestSurvival()
        rfc.fit(lung_X, lung_y)
        assert rfc.regularization_factor_ == []
        assert not rfc.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            rfc = RangerForestSurvival(regularization_factor=r)
            with pytest.raises(ValueError):
                rfc.fit(lung_X, lung_y)

        # vector of ones isn't applied
        rfc = RangerForestSurvival(regularization_factor=[1] * lung_X.shape[1])
        rfc.fit(lung_X, lung_y)
        assert rfc.regularization_factor_ == []
        assert not rfc.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        rfc = RangerForestSurvival(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            rfc.fit(lung_X, lung_y)
        assert rfc.n_jobs_ == 1
        assert rfc.regularization_factor_ == reg
        assert rfc.use_regularization_factor_

    def test_always_split_features(self, lung_X, lung_y):
        rfc = RangerForestSurvival(always_split_features=[0])
        rfc.fit(lung_X, lung_y)
        # feature 0 is in every tree split
        for tree in rfc.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree
