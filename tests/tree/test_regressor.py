import pickle
import random
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import csr_matrix
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skranger.tree.regressor import RangerTreeRegressor


class TestRangerTreeRegressor:
    def test_init(self):
        _ = RangerTreeRegressor()

    def test_fit(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(wine_X, wine_y)
        check_is_fitted(tree)
        assert hasattr(tree, "ranger_forest_")
        assert hasattr(tree, "n_features_in_")

    def test_predict(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        pred = tree.predict(wine_X)
        assert len(pred) == wine_X.shape[0]

        # test with single record
        wine_X_record = wine_X[0:1, :]
        pred = tree.predict(wine_X_record)
        assert len(pred) == 1

    def test_serialize(self, wine_X, wine_y):
        tf = tempfile.TemporaryFile()
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(wine_X)
        assert len(pred) == wine_X.shape[0]

    def test_clone(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        clone(tree)

    def test_verbose(self, wine_X, wine_y, verbose, capfd):
        tree = RangerTreeRegressor(verbose=verbose)
        tree.fit(wine_X, wine_y)
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
        tree = RangerTreeRegressor(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)
            return

        tree.fit(wine_X, wine_y)
        if importance == "none":
            assert tree.importance_mode_ == 0
        elif importance == "impurity":
            assert tree.importance_mode_ == 1
        elif importance == "impurity_corrected":
            assert tree.importance_mode_ == 5
        elif importance == "permutation":
            if local_importance:
                assert tree.importance_mode_ == 6
            elif scale_permutation_importance:
                assert tree.importance_mode_ == 2
            else:
                assert tree.importance_mode_ == 3

    def test_mtry(self, wine_X, wine_y, mtry):
        tree = RangerTreeRegressor(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > wine_X.shape[0]):
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)
            return

        tree.fit(wine_X, wine_y)
        if callable(mtry):
            assert tree.mtry_ == mtry(wine_X.shape[1])
        else:
            assert tree.mtry_ == mtry

    def test_inbag(self, wine_X, wine_y):
        inbag = [[1, 2, 3]]
        tree = RangerTreeRegressor(inbag=inbag)
        tree.fit(wine_X, wine_y)

        # can't use inbag with sample weight
        tree = RangerTreeRegressor(inbag=inbag)
        with pytest.raises(ValueError):
            tree.fit(wine_X, wine_y, sample_weight=[1] * len(wine_y))

        # can't use class sampling and inbag
        tree = RangerTreeRegressor(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            tree.fit(wine_X, wine_y)

    def test_sample_fraction(self, wine_X, wine_y):
        tree = RangerTreeRegressor(sample_fraction=0.69)
        tree.fit(wine_X, wine_y)
        assert tree.sample_fraction_ == [0.69]

        # test with single record
        wine_X_record = wine_X[0:1, :]
        pred = tree.predict(wine_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, wine_X, wine_y, replace):
        tree = RangerTreeRegressor(replace=replace)
        tree.fit(wine_X, wine_y)

        if replace:
            assert tree.sample_fraction_ == [1.0]
        else:
            assert tree.sample_fraction_ == [0.632]

    def test_categorical_features(self, wine_X, wine_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(wine_X.shape[0])])
        )
        wine_X_c = np.hstack((wine_X, categorical_col.transpose()))
        categorical_features = [wine_X.shape[1]]

        tree = RangerTreeRegressor(
            respect_categorical_features=respect_categorical_features,
            categorical_features=categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                tree.fit(wine_X_c, wine_y)
            return

        tree.fit(wine_X_c, wine_y)
        tree.predict(wine_X_c)

    def test_split_rule(self, wine_X, wine_y, split_rule):
        tree = RangerTreeRegressor(split_rule=split_rule)
        assert tree.criterion == split_rule

        if split_rule not in ["variance", "extratrees", "maxstat", "beta"]:
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)
            return

        # beta can only be used with targets between 0 and 1
        if split_rule == "beta":
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)

        wine_01 = [0.5 for _ in wine_y]
        tree.fit(wine_X, wine_01)

        if split_rule == "variance":
            assert tree.split_rule_ == 1
        elif split_rule == "extratrees":
            assert tree.split_rule_ == 5
        elif split_rule == "maxstat":
            assert tree.split_rule_ == 4
        elif split_rule == "beta":
            assert tree.split_rule_ == 6

        if split_rule == "extratrees":
            tree = RangerTreeRegressor(
                split_rule=split_rule,
                respect_categorical_features="partition",
                save_memory=True,
            )
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)
        else:
            tree = RangerTreeRegressor(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)

    def test_split_select_weights(self, wine_X, wine_y):
        n_trees = 1
        weights = [0.1] * wine_X.shape[1]
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y, split_select_weights=weights)

        weights = [0.1] * (wine_X.shape[1] - 1)
        tree = RangerTreeRegressor()

        with pytest.raises(RuntimeError):
            tree.fit(wine_X, wine_y, split_select_weights=weights)

        weights = [[0.1] * (wine_X.shape[1])] * n_trees
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y, split_select_weights=weights)

        weights = [[0.1] * (wine_X.shape[1])] * (n_trees + 1)
        tree = RangerTreeRegressor()
        with pytest.raises(RuntimeError):
            tree.fit(wine_X, wine_y, split_select_weights=weights)

    def test_regularization(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        assert tree.regularization_factor_ == []
        assert not tree.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            tree = RangerTreeRegressor(regularization_factor=r)
            with pytest.raises(ValueError):
                tree.fit(wine_X, wine_y)

        # vector of ones isn't applied
        tree = RangerTreeRegressor(regularization_factor=[1] * wine_X.shape[1])
        tree.fit(wine_X, wine_y)
        assert tree.regularization_factor_ == []
        assert not tree.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        tree = RangerTreeRegressor(regularization_factor=reg)
        tree.fit(wine_X, wine_y)
        assert tree.regularization_factor_ == reg
        assert tree.use_regularization_factor_

    def test_always_split_features(self, wine_X, wine_y):
        tree = RangerTreeRegressor(always_split_features=[0])
        tree.fit(wine_X, wine_y)
        # feature 0 is in every tree split
        for tree in tree.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_check_estimator(self):
        check_estimator(RangerTreeRegressor())

    def test_get_depth(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        leaves = tree.apply(wine_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(wine_X)

    def test_decision_path(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        paths = tree.decision_path(wine_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(wine_X)

    def test_tree_interface(self, wine_X, wine_y):
        tree = RangerTreeRegressor()
        tree.fit(wine_X, wine_y)
        # access attributes the way we would expect to in sklearn
        tree_ = tree.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        feature = tree_.feature
        threshold = tree_.threshold
        max_depth = tree_.max_depth
        n_node_samples = tree_.n_node_samples
        weighted_n_node_samples = tree_.weighted_n_node_samples
        node_count = tree_.node_count
        capacity = tree_.capacity
        n_outputs = tree_.n_outputs
        n_classes = tree_.n_classes
        value = tree_.value
        assert value.shape == (node_count, 1, 1)
