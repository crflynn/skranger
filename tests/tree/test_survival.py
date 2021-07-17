import pickle
import random
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree._tree import csr_matrix
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skranger.tree import RangerTreeSurvival


class TestRangerTreeSurvival:
    def test_init(self):
        _ = RangerTreeSurvival()

    def test_fit(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(lung_X, lung_y)
        check_is_fitted(tree)
        assert hasattr(tree, "event_times_")
        assert hasattr(tree, "cumulative_hazard_function_")
        assert hasattr(tree, "ranger_forest_")
        assert hasattr(tree, "n_features_in_")

    def test_predict(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        pred = tree.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

        # test with single record
        lung_X_record = lung_X.values[0:1, :]
        pred = tree.predict(lung_X_record)
        assert len(pred) == 1

    def test_predict_cumulative_hazard_function(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        pred = tree.predict_cumulative_hazard_function(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_predict_survival_function(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        pred = tree.predict_survival_function(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        tf = tempfile.TemporaryFile()
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        clone(tree)

    def test_verbose(self, lung_X, lung_y, verbose, capfd):
        tree = RangerTreeSurvival(verbose=verbose)
        tree.fit(lung_X, lung_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(
        self, lung_X, lung_y, importance, scale_permutation_importance, local_importance
    ):
        tree = RangerTreeSurvival(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)
            return

        tree.fit(lung_X, lung_y)
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

    def test_mtry(self, lung_X, lung_y, mtry):
        tree = RangerTreeSurvival(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > lung_X.shape[0]):
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)
            return

        tree.fit(lung_X, lung_y)
        if callable(mtry):
            assert tree.mtry_ == mtry(lung_X.shape[1])
        else:
            assert tree.mtry_ == mtry

    def test_inbag(self, lung_X, lung_y):
        inbag = [[1, 2, 3]]
        tree = RangerTreeSurvival(inbag=inbag)
        tree.fit(lung_X, lung_y)

        # can't use inbag with sample weight
        tree = RangerTreeSurvival(inbag=inbag)
        with pytest.raises(ValueError):
            tree.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))

        # can't use class sampling and inbag
        tree = RangerTreeSurvival(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            tree.fit(lung_X, lung_y)

    def test_sample_fraction(self, lung_X, lung_y):
        tree = RangerTreeSurvival(sample_fraction=0.69)
        tree.fit(lung_X, lung_y)
        assert tree.sample_fraction_ == [0.69]

        # test with single record
        lung_X_record = lung_X.values[0:1, :]
        pred = tree.predict(lung_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, lung_X, lung_y, replace):
        tree = RangerTreeSurvival(replace=replace)
        tree.fit(lung_X, lung_y)

        if replace:
            assert tree.sample_fraction_ == [1.0]
        else:
            assert tree.sample_fraction_ == [0.632]

    def test_categorical_features(self, lung_X, lung_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(lung_X.shape[0])])
        )
        lung_X_c = np.hstack((lung_X, categorical_col.transpose()))
        categorical_features = [lung_X.shape[1]]

        tree = RangerTreeSurvival(
            respect_categorical_features=respect_categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                tree.fit(lung_X_c, lung_y, categorical_features=categorical_features)
            return

        tree.fit(lung_X_c, lung_y, categorical_features=categorical_features)
        tree.predict(lung_X_c)

    def test_split_rule(self, lung_X, lung_y, split_rule):
        tree = RangerTreeSurvival(split_rule=split_rule)

        if split_rule not in ["logrank", "extratrees", "C", "C_ignore_ties", "maxstat"]:
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)
            return

        tree.fit(lung_X, lung_y)

        if split_rule == "logrank":
            assert tree.split_rule_ == 1
        elif split_rule == "extratrees":
            assert tree.split_rule_ == 5
        elif split_rule == "C":
            assert tree.split_rule_ == 2
        elif split_rule == "C_ignore_ties":
            assert tree.split_rule_ == 3
        elif split_rule == "maxstat":
            assert tree.split_rule_ == 4

        if split_rule != "extratrees":
            tree = RangerTreeSurvival(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)

    def test_split_select_weights(self, lung_X, lung_y):
        n_trees = 1
        weights = [0.1] * lung_X.shape[1]
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y, split_select_weights=weights)

        weights = [0.1] * (lung_X.shape[1] - 1)
        tree = RangerTreeSurvival()

        with pytest.raises(RuntimeError):
            tree.fit(lung_X, lung_y, split_select_weights=weights)

        weights = [[0.1] * (lung_X.shape[1])] * n_trees
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y, split_select_weights=weights)

        weights = [[0.1] * (lung_X.shape[1])] * (n_trees + 1)
        tree = RangerTreeSurvival()
        with pytest.raises(RuntimeError):
            tree.fit(lung_X, lung_y, split_select_weights=weights)

    def test_regularization(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        assert tree.regularization_factor_ == []
        assert not tree.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            tree = RangerTreeSurvival(regularization_factor=r)
            with pytest.raises(ValueError):
                tree.fit(lung_X, lung_y)

        # vector of ones isn't applied
        tree = RangerTreeSurvival(regularization_factor=[1] * lung_X.shape[1])
        tree.fit(lung_X, lung_y)
        assert tree.regularization_factor_ == []
        assert not tree.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        tree = RangerTreeSurvival(regularization_factor=reg,)
        tree.fit(lung_X, lung_y)
        assert tree.regularization_factor_ == reg
        assert tree.use_regularization_factor_

    def test_always_split_features(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y, always_split_features=[0])
        # feature 0 is in every tree split
        for tree in tree.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_sample_weight(self, lung_X, lung_y):
        forest_w = RangerTreeSurvival()
        forest_w.fit(lung_X, lung_y, sample_weight=[1] * len(lung_y))
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)

        pred_w = forest_w.predict(lung_X)
        pred = tree.predict(lung_X)

        np.testing.assert_array_equal(pred.reshape(-1, 1), pred_w.reshape(-1, 1))

    def test_get_tags(self):
        tree = RangerTreeSurvival()
        tags = tree._get_tags()
        assert tags["requires_y"]

    # We can't check this because we conform to scikit-survival api,
    # rather than scikit-learn's
    # def test_check_estimator(self):
    #     check_estimator(RangerTreeSurvival())

    def test_get_depth(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        leaves = tree.apply(lung_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(lung_X)

    def test_decision_path(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
        paths = tree.decision_path(lung_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(lung_X)

    def test_tree_interface(self, lung_X, lung_y):
        tree = RangerTreeSurvival()
        tree.fit(lung_X, lung_y)
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
        # value = tree_.value  # FIXME
