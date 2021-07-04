import pickle
import random
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skranger.ensemble import RangerForestRegressor


class TestRangerForestRegressor:
    def test_init(self):
        _ = RangerForestRegressor()

    def test_fit(self, boston_X, boston_y):
        rfr = RangerForestRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(rfr)
        rfr.fit(boston_X, boston_y)
        check_is_fitted(rfr)
        assert hasattr(rfr, "ranger_forest_")
        assert hasattr(rfr, "n_features_in_")

    def test_predict(self, boston_X, boston_y):
        rfr = RangerForestRegressor()
        rfr.fit(boston_X, boston_y)
        pred = rfr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

        # test with single record
        boston_X_record = boston_X[0:1, :]
        pred = rfr.predict(boston_X_record)
        assert len(pred) == 1

    def test_serialize(self, boston_X, boston_y):
        tf = tempfile.TemporaryFile()
        rfr = RangerForestRegressor()
        rfr.fit(boston_X, boston_y)
        pickle.dump(rfr, tf)
        tf.seek(0)
        new_rfr = pickle.load(tf)
        pred = new_rfr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

    def test_clone(self, boston_X, boston_y):
        rfr = RangerForestRegressor()
        rfr.fit(boston_X, boston_y)
        clone(rfr)

    def test_verbose(self, boston_X, boston_y, verbose, capfd):
        rfr = RangerForestRegressor(verbose=verbose)
        rfr.fit(boston_X, boston_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(
        self,
        boston_X,
        boston_y,
        importance,
        scale_permutation_importance,
        local_importance,
    ):
        rfr = RangerForestRegressor(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)
            return

        rfr.fit(boston_X, boston_y)
        if importance == "none":
            assert rfr.importance_mode_ == 0
        elif importance == "impurity":
            assert rfr.importance_mode_ == 1
        elif importance == "impurity_corrected":
            assert rfr.importance_mode_ == 5
        elif importance == "permutation":
            if local_importance:
                assert rfr.importance_mode_ == 6
            elif scale_permutation_importance:
                assert rfr.importance_mode_ == 2
            else:
                assert rfr.importance_mode_ == 3

    def test_importance_pvalues(self, boston_X_mod, boston_y, importance, mod):
        rfc = RangerForestRegressor(importance=importance)
        np.random.seed(42)

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfc.fit(boston_X_mod, boston_y)
            return

        if not importance == "impurity_corrected":
            rfc.fit(boston_X_mod, boston_y)
            with pytest.raises(ValueError):
                rfc.get_importance_pvalues()
            return

        # Test error for no non-negative importance values

        if mod == "none":
            rfc.fit(boston_X_mod, boston_y)
            with pytest.raises(ValueError):
                rfc.get_importance_pvalues()
            return

        rfc.fit(boston_X_mod, boston_y)
        assert len(rfc.get_importance_pvalues()) == boston_X_mod.shape[1]

    def test_mtry(self, boston_X, boston_y, mtry):
        rfr = RangerForestRegressor(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > boston_X.shape[0]):
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)
            return

        rfr.fit(boston_X, boston_y)
        if callable(mtry):
            assert rfr.mtry_ == mtry(boston_X.shape[1])
        else:
            assert rfr.mtry_ == mtry

    def test_inbag(self, boston_X, boston_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        rfr = RangerForestRegressor(n_estimators=2, inbag=inbag)
        rfr.fit(boston_X, boston_y)

        # inbag list different length from n_estimators
        rfr = RangerForestRegressor(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            rfr.fit(boston_X, boston_y)

        # can't use inbag with sample weight
        rfr = RangerForestRegressor(inbag=inbag)
        with pytest.raises(ValueError):
            rfr.fit(boston_X, boston_y, sample_weight=[1] * len(boston_y))

        # can't use class sampling and inbag
        rfr = RangerForestRegressor(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            rfr.fit(boston_X, boston_y)

    def test_sample_fraction(self, boston_X, boston_y):
        rfr = RangerForestRegressor(sample_fraction=0.69)
        rfr.fit(boston_X, boston_y)
        assert rfr.sample_fraction_ == [0.69]

        # test with single record
        boston_X_record = boston_X[0:1, :]
        pred = rfr.predict(boston_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, boston_X, boston_y, replace):
        rfr = RangerForestRegressor(replace=replace)
        rfr.fit(boston_X, boston_y)

        if replace:
            assert rfr.sample_fraction_ == [1.0]
        else:
            assert rfr.sample_fraction_ == [0.632]

    def test_categorical_features(
        self, boston_X, boston_y, respect_categorical_features
    ):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(boston_X.shape[0])])
        )
        boston_X_c = np.hstack((boston_X, categorical_col.transpose()))
        categorical_features = [boston_X.shape[1]]

        rfr = RangerForestRegressor(
            respect_categorical_features=respect_categorical_features,
            categorical_features=categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                rfr.fit(boston_X_c, boston_y)
            return

        rfr.fit(boston_X_c, boston_y)

    def test_split_rule(self, boston_X, boston_y, split_rule):
        rfr = RangerForestRegressor(split_rule=split_rule)

        if split_rule not in ["variance", "extratrees", "maxstat", "beta"]:
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)
            return

        # beta can only be used with targets between 0 and 1
        if split_rule == "beta":
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)

        boston_01 = [0.5 for _ in boston_y]
        rfr.fit(boston_X, boston_01)

        if split_rule == "variance":
            assert rfr.split_rule_ == 1
        elif split_rule == "extratrees":
            assert rfr.split_rule_ == 5
        elif split_rule == "maxstat":
            assert rfr.split_rule_ == 4
        elif split_rule == "beta":
            assert rfr.split_rule_ == 6

        if split_rule == "extratrees":
            rfr = RangerForestRegressor(
                split_rule=split_rule,
                respect_categorical_features="partition",
                save_memory=True,
            )
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)
        else:
            rfr = RangerForestRegressor(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)

    def test_split_select_weights(self, boston_X, boston_y):
        n_trees = 10
        weights = [0.1] * boston_X.shape[1]
        rfr = RangerForestRegressor(n_estimators=n_trees)
        rfr.fit(boston_X, boston_y, split_select_weights=weights)

        weights = [0.1] * (boston_X.shape[1] - 1)
        rfr = RangerForestRegressor(n_estimators=n_trees)

        with pytest.raises(RuntimeError):
            rfr.fit(boston_X, boston_y, split_select_weights=weights)

        weights = [[0.1] * (boston_X.shape[1])] * n_trees
        rfr = RangerForestRegressor(n_estimators=n_trees)
        rfr.fit(boston_X, boston_y, split_select_weights=weights)

        weights = [[0.1] * (boston_X.shape[1])] * (n_trees + 1)
        rfr = RangerForestRegressor(n_estimators=n_trees)
        with pytest.raises(RuntimeError):
            rfr.fit(boston_X, boston_y, split_select_weights=weights)

    def test_regularization(self, boston_X, boston_y):
        rfr = RangerForestRegressor()
        rfr.fit(boston_X, boston_y)
        assert rfr.regularization_factor_ == []
        assert not rfr.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            rfr = RangerForestRegressor(regularization_factor=r)
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)

        # vector of ones isn't applied
        rfr = RangerForestRegressor(regularization_factor=[1] * boston_X.shape[1])
        rfr.fit(boston_X, boston_y)
        assert rfr.regularization_factor_ == []
        assert not rfr.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        rfr = RangerForestRegressor(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            rfr.fit(boston_X, boston_y)
        assert rfr.n_jobs_ == 1
        assert rfr.regularization_factor_ == reg
        assert rfr.use_regularization_factor_

    def test_always_split_features(self, boston_X, boston_y):
        rfr = RangerForestRegressor(always_split_features=[0])
        rfr.fit(boston_X, boston_y)
        # feature 0 is in every tree split
        for tree in rfr.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_quantile_regression(self, boston_X, boston_y):
        X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y)
        rfr = RangerForestRegressor(quantiles=False)
        rfr.fit(X_train, y_train)
        assert not hasattr(rfr, "random_node_values_")
        with pytest.raises(ValueError):
            rfr.predict_quantiles(X_test)
        rfr = RangerForestRegressor(quantiles=True)
        rfr.fit(X_train, y_train)
        assert hasattr(rfr, "random_node_values_")
        quantiles_lower = rfr.predict_quantiles(X_test, quantiles=[0.1])
        quantiles_upper = rfr.predict_quantiles(X_test, quantiles=[0.9])
        assert np.less(quantiles_lower, quantiles_upper).all()
        assert quantiles_upper.ndim == 1
        quantiles = rfr.predict_quantiles(X_test, quantiles=[0.1, 0.9])
        assert quantiles.ndim == 2

    def test_feature_importances_(
        self, boston_X, boston_y, importance, local_importance
    ):
        rfr = RangerForestRegressor(
            importance=importance, local_importance=local_importance
        )
        with pytest.raises(AttributeError):
            _ = rfr.feature_importances_

        if importance == "INVALID":
            with pytest.raises(ValueError):
                rfr.fit(boston_X, boston_y)
            return

        rfr.fit(boston_X, boston_y)
        if importance == "none":
            with pytest.raises(ValueError):
                _ = rfr.feature_importances_
        else:
            assert len(rfr.feature_importances_) == boston_X.shape[1]

    def test_check_estimator(self):
        check_estimator(RangerForestRegressor())
