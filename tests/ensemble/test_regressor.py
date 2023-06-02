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
from skranger.tree import RangerTreeRegressor


class TestRangerForestRegressor:
    def test_init(self):
        _ = RangerForestRegressor()

    def test_fit(self, wine_X, wine_y):
        forest = RangerForestRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(wine_X, wine_y)
        check_is_fitted(forest)
        assert hasattr(forest, "ranger_forest_")
        assert hasattr(forest, "n_features_in_")

    def test_predict(self, wine_X, wine_y):
        forest = RangerForestRegressor()
        forest.fit(wine_X, wine_y)
        pred = forest.predict(wine_X)
        assert len(pred) == wine_X.shape[0]

        # test with single record
        wine_X_record = wine_X[0:1, :]
        pred = forest.predict(wine_X_record)
        assert len(pred) == 1

    def test_serialize(self, wine_X, wine_y):
        tf = tempfile.TemporaryFile()
        forest = RangerForestRegressor()
        forest.fit(wine_X, wine_y)
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(wine_X)
        assert len(pred) == wine_X.shape[0]

    def test_clone(self, wine_X, wine_y):
        forest = RangerForestRegressor()
        forest.fit(wine_X, wine_y)
        clone(forest)

    def test_verbose(self, wine_X, wine_y, verbose, capfd):
        forest = RangerForestRegressor(verbose=verbose)
        forest.fit(wine_X, wine_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(
        self,
        wine_X,
        wine_y,
        importance,
        scale_permutation_importance,
        local_importance,
    ):
        forest = RangerForestRegressor(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)
            return

        forest.fit(wine_X, wine_y)
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

    def test_importance_pvalues(self, wine_X_mod, wine_y, importance, mod):
        rfc = RangerForestRegressor(importance=importance)
        np.random.seed(42)

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfc.fit(wine_X_mod, wine_y)
            return

        if not importance == "impurity_corrected":
            rfc.fit(wine_X_mod, wine_y)
            with pytest.raises(ValueError):
                rfc.get_importance_pvalues()
            return

        # Test error for no non-negative importance values

        if mod == "none":
            rfc.fit(wine_X_mod, wine_y)
            with pytest.raises(ValueError):
                rfc.get_importance_pvalues()
            return

        rfc.fit(wine_X_mod, wine_y)
        assert len(rfc.get_importance_pvalues()) == wine_X_mod.shape[1]

    def test_mtry(self, wine_X, wine_y, mtry):
        forest = RangerForestRegressor(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > wine_X.shape[0]):
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)
            return

        forest.fit(wine_X, wine_y)
        if callable(mtry):
            assert forest.mtry_ == mtry(wine_X.shape[1])
        else:
            assert forest.mtry_ == mtry

    def test_inbag(self, wine_X, wine_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        forest = RangerForestRegressor(n_estimators=2, inbag=inbag)
        forest.fit(wine_X, wine_y)

        # inbag list different length from n_estimators
        forest = RangerForestRegressor(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            forest.fit(wine_X, wine_y)

        # can't use inbag with sample weight
        forest = RangerForestRegressor(inbag=inbag)
        with pytest.raises(ValueError):
            forest.fit(wine_X, wine_y, sample_weight=[1] * len(wine_y))

        # can't use class sampling and inbag
        forest = RangerForestRegressor(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            forest.fit(wine_X, wine_y)

    def test_sample_fraction(self, wine_X, wine_y):
        forest = RangerForestRegressor(sample_fraction=0.69)
        forest.fit(wine_X, wine_y)
        assert forest.sample_fraction_ == [0.69]

        # test with single record
        wine_X_record = wine_X[0:1, :]
        pred = forest.predict(wine_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, wine_X, wine_y, replace):
        forest = RangerForestRegressor(replace=replace)
        forest.fit(wine_X, wine_y)

        if replace:
            assert forest.sample_fraction_ == [1.0]
        else:
            assert forest.sample_fraction_ == [0.632]

    def test_categorical_features(self, wine_X, wine_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(wine_X.shape[0])])
        )
        wine_X_c = np.hstack((wine_X, categorical_col.transpose()))
        categorical_features = [wine_X.shape[1]]

        forest = RangerForestRegressor(
            respect_categorical_features=respect_categorical_features,
            categorical_features=categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                forest.fit(wine_X_c, wine_y)
            return

        forest.fit(wine_X_c, wine_y)
        forest.predict(wine_X_c)

    def test_split_rule(self, wine_X, wine_y, split_rule):
        forest = RangerForestRegressor(split_rule=split_rule)
        assert forest.criterion == split_rule

        if split_rule not in ["variance", "extratrees", "maxstat", "beta"]:
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)
            return

        # beta can only be used with targets between 0 and 1
        if split_rule == "beta":
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)

        wine_01 = [0.5 for _ in wine_y]
        forest.fit(wine_X, wine_01)

        if split_rule == "variance":
            assert forest.split_rule_ == 1
        elif split_rule == "extratrees":
            assert forest.split_rule_ == 5
        elif split_rule == "maxstat":
            assert forest.split_rule_ == 4
        elif split_rule == "beta":
            assert forest.split_rule_ == 6

        if split_rule == "extratrees":
            forest = RangerForestRegressor(
                split_rule=split_rule,
                respect_categorical_features="partition",
                save_memory=True,
            )
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)
        else:
            forest = RangerForestRegressor(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)

    def test_split_select_weights(self, wine_X, wine_y):
        n_trees = 10
        weights = [0.1] * wine_X.shape[1]
        forest = RangerForestRegressor(n_estimators=n_trees)
        forest.fit(wine_X, wine_y, split_select_weights=weights)

        weights = [0.1] * (wine_X.shape[1] - 1)
        forest = RangerForestRegressor(n_estimators=n_trees)

        with pytest.raises(RuntimeError):
            forest.fit(wine_X, wine_y, split_select_weights=weights)

        weights = [[0.1] * (wine_X.shape[1])] * n_trees
        forest = RangerForestRegressor(n_estimators=n_trees)
        forest.fit(wine_X, wine_y, split_select_weights=weights)

        weights = [[0.1] * (wine_X.shape[1])] * (n_trees + 1)
        forest = RangerForestRegressor(n_estimators=n_trees)
        with pytest.raises(RuntimeError):
            forest.fit(wine_X, wine_y, split_select_weights=weights)

    def test_regularization(self, wine_X, wine_y):
        forest = RangerForestRegressor()
        forest.fit(wine_X, wine_y)
        assert forest.regularization_factor_ == []
        assert not forest.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            forest = RangerForestRegressor(regularization_factor=r)
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)

        # vector of ones isn't applied
        forest = RangerForestRegressor(regularization_factor=[1] * wine_X.shape[1])
        forest.fit(wine_X, wine_y)
        assert forest.regularization_factor_ == []
        assert not forest.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        forest = RangerForestRegressor(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            forest.fit(wine_X, wine_y)
        assert forest.n_jobs_ == 1
        assert forest.regularization_factor_ == reg
        assert forest.use_regularization_factor_

    def test_always_split_features(self, wine_X, wine_y):
        forest = RangerForestRegressor(always_split_features=[0])
        forest.fit(wine_X, wine_y)
        # feature 0 is in every tree split
        for tree in forest.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_quantile_regression(self, wine_X, wine_y):
        X_train, X_test, y_train, y_test = train_test_split(wine_X, wine_y)
        forest = RangerForestRegressor(quantiles=False)
        forest.fit(X_train, y_train)
        assert not hasattr(forest, "random_node_values_")
        with pytest.raises(ValueError):
            forest.predict_quantiles(X_test, quantiles=[0.2, 0.5, 0.8])
        forest = RangerForestRegressor(quantiles=True)
        forest.fit(X_train, y_train)
        assert hasattr(forest, "random_node_values_")
        quantiles_lower = forest.predict_quantiles(X_test, quantiles=[0.1])
        quantiles_upper = forest.predict_quantiles(X_test, quantiles=[0.9])
        assert np.less_equal(quantiles_lower, quantiles_upper).all()
        assert quantiles_upper.ndim == 1
        quantiles = forest.predict_quantiles(X_test, quantiles=[0.1, 0.9])
        assert quantiles.shape == (X_test.shape[0], 2)
        assert np.sum(np.isnan(quantiles_lower)) == 0
        assert np.sum(np.isnan(quantiles_upper)) == 0

        # test predict method
        pred = forest.predict(X_test, quantiles=[0.2, 0.5])
        assert pred.shape == (X_test.shape[0], 2)
        assert np.sum(np.isnan(pred)) == 0
        pred = forest.predict(X_test, quantiles=[0.2])
        assert pred.ndim == 1
        assert np.sum(np.isnan(pred)) == 0

        # test with single record
        wine_X_record = wine_X[0:1, :]
        pred = forest.predict(wine_X_record, quantiles=[0.2, 0.5])
        assert pred.shape == (1, 2)
        assert np.sum(np.isnan(pred)) == 0

    def test_feature_importances_(self, wine_X, wine_y, importance, local_importance):
        forest = RangerForestRegressor(
            importance=importance, local_importance=local_importance
        )
        with pytest.raises(AttributeError):
            _ = forest.feature_importances_

        if importance == "INVALID":
            with pytest.raises(ValueError):
                forest.fit(wine_X, wine_y)
            return

        forest.fit(wine_X, wine_y)
        if importance == "none":
            with pytest.raises(ValueError):
                _ = forest.feature_importances_
        else:
            assert len(forest.feature_importances_) == wine_X.shape[1]

    def test_estimators_(self, wine_X, wine_y):
        forest = RangerForestRegressor(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(wine_X, wine_y)
        with pytest.raises(ValueError):
            _ = forest.estimators_
        forest = RangerForestRegressor(n_estimators=10, enable_tree_details=True)
        forest.fit(wine_X, wine_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], RangerTreeRegressor)
        check_is_fitted(estimators[0])

    def test_get_estimator(self, wine_X, wine_y):
        forest = RangerForestRegressor(n_estimators=10)
        with pytest.raises(NotFittedError):
            _ = forest.get_estimator(idx=0)
        forest.fit(wine_X, wine_y)
        with pytest.raises(ValueError):
            _ = forest.get_estimator(0)
        forest = RangerForestRegressor(n_estimators=10, enable_tree_details=True)
        forest.fit(wine_X, wine_y)
        estimator = forest.get_estimator(0)
        estimator.predict(wine_X)
        assert isinstance(estimator, RangerTreeRegressor)
        with pytest.raises(IndexError):
            _ = forest.get_estimator(idx=20)

    def test_check_estimator(self):
        check_estimator(RangerForestRegressor())
