import pickle
import random
import tempfile

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import csr_matrix
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skranger.tree import RangerTreeClassifier


class TestRangerTreeClassifier:
    def test_init(self):
        _ = RangerTreeClassifier()

    def test_fit(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        with pytest.raises(NotFittedError):
            check_is_fitted(tree)
        tree.fit(iris_X, iris_y)
        check_is_fitted(tree)
        assert hasattr(tree, "classes_")
        assert hasattr(tree, "n_classes_")
        assert hasattr(tree, "ranger_forest_")
        assert hasattr(tree, "ranger_class_order_")
        assert hasattr(tree, "n_features_in_")

    def test_predict(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        pred = tree.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = tree.predict(iris_X_record)
        assert len(pred) == 1

    def test_predict_proba(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        pred = tree.predict_proba(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = tree.predict_proba(iris_X_record)
        assert len(pred) == 1

    def test_predict_log_proba(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        pred = tree.predict_log_proba(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = tree.predict_log_proba(iris_X_record)
        assert len(pred) == 1

    def test_serialize(self, iris_X, iris_y):
        tf = tempfile.TemporaryFile()
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        pickle.dump(tree, tf)
        tf.seek(0)
        new_tree = pickle.load(tf)
        pred = new_tree.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_clone(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        clone(tree)

    def test_verbose(self, iris_X, iris_y, verbose, capfd):
        tree = RangerTreeClassifier(verbose=verbose)
        tree.fit(iris_X, iris_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(
        self, iris_X, iris_y, importance, scale_permutation_importance, local_importance
    ):
        tree = RangerTreeClassifier(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
            return

        tree.fit(iris_X, iris_y)
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

    def test_mtry(self, iris_X, iris_y, mtry):
        tree = RangerTreeClassifier(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > iris_X.shape[0]):
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
            return

        tree.fit(iris_X, iris_y)
        if callable(mtry):
            assert tree.mtry_ == mtry(iris_X.shape[1])
        else:
            assert tree.mtry_ == mtry

    def test_inbag(self, iris_X, iris_y):
        inbag = [[1, 2, 3]]
        tree = RangerTreeClassifier(inbag=inbag)
        tree.fit(iris_X, iris_y)

        # can't use inbag with sample weight
        tree = RangerTreeClassifier(inbag=inbag)
        with pytest.raises(ValueError):
            tree.fit(iris_X, iris_y, sample_weight=[1] * len(iris_y))

        # can't use class sampling and inbag
        tree = RangerTreeClassifier(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            tree.fit(iris_X, iris_y)

    def test_sample_fraction(self, iris_X, iris_y):
        tree = RangerTreeClassifier(sample_fraction=[0.69])
        tree.fit(iris_X, iris_y)
        assert tree.sample_fraction_ == [0.69]
        tree = RangerTreeClassifier(sample_fraction=0.69)
        tree.fit(iris_X, iris_y)
        assert tree.sample_fraction_ == [0.69]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = tree.predict(iris_X_record)
        assert len(pred) == 1
        pred = tree.predict_proba(iris_X_record)
        assert len(pred) == 1
        pred = tree.predict_log_proba(iris_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, iris_X, iris_y, replace):
        tree = RangerTreeClassifier(replace=replace)
        tree.fit(iris_X, iris_y)

        if replace:
            assert tree.sample_fraction_ == [1.0]
        else:
            assert tree.sample_fraction_ == [0.632]

    def test_categorical_features(self, iris_X, iris_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(iris_X.shape[0])])
        )
        iris_X_c = np.hstack((iris_X, categorical_col.transpose()))
        categorical_features = [iris_X.shape[1]]

        tree = RangerTreeClassifier(
            respect_categorical_features=respect_categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                tree.fit(iris_X_c, iris_y, categorical_features=categorical_features)
            return

        tree.fit(iris_X_c, iris_y, categorical_features=categorical_features)
        tree.predict(iris_X_c)

    def test_split_rule(self, iris_X, iris_y, split_rule):
        tree = RangerTreeClassifier(split_rule=split_rule)

        if split_rule not in ["gini", "extratrees", "hellinger"]:
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
            return

        # hellinger can only be used in binary classification
        if split_rule == "hellinger":
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)

        iris_2 = [0 if v == 2 else v for v in iris_y]
        tree.fit(iris_X, iris_2)

        if split_rule == "gini":
            assert tree.split_rule_ == 1
        elif split_rule == "extratrees":
            assert tree.split_rule_ == 5
        if split_rule == "hellinger":
            assert tree.split_rule_ == 7

        if split_rule == "extratrees":
            tree = RangerTreeClassifier(
                split_rule=split_rule,
                respect_categorical_features="partition",
                save_memory=True,
            )
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)
        else:
            tree = RangerTreeClassifier(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)

    def test_class_weights(self, iris_X, iris_y):
        X_train, X_test, y_train, y_test = train_test_split(
            iris_X, iris_y, test_size=0.5, random_state=42
        )
        tree = RangerTreeClassifier()
        weights = {
            0: 0.7,
            1: 0.2,
            2: 0.1,
        }
        tree.fit(X_train, y_train, class_weights=weights)
        tree.predict(X_test)

        tree = RangerTreeClassifier()
        m = {0: "a", 1: "b", 2: "c"}
        y_train_str = [m.get(v) for v in y_train]
        weights = {
            "a": 0.7,
            "b": 0.2,
            "c": 0.1,
        }
        tree.fit(X_train, y_train_str, class_weights=weights)
        tree.predict(X_test)

        weights = {
            0: 0.7,
        }
        with pytest.raises(ValueError):
            tree.fit(X_train, y_train, class_weights=weights)

    def test_split_select_weights(self, iris_X, iris_y):
        n_trees = 1
        weights = [0.1] * iris_X.shape[1]
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y, split_select_weights=weights)

        weights = [0.1] * (iris_X.shape[1] - 1)
        tree = RangerTreeClassifier()

        with pytest.raises(RuntimeError):
            tree.fit(iris_X, iris_y, split_select_weights=weights)

        weights = [[0.1] * (iris_X.shape[1])] * n_trees
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y, split_select_weights=weights)

        weights = [[0.1] * (iris_X.shape[1])] * (n_trees + 1)
        tree = RangerTreeClassifier()
        with pytest.raises(RuntimeError):
            tree.fit(iris_X, iris_y, split_select_weights=weights)

    def test_regularization(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        assert tree.regularization_factor_ == []
        assert not tree.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            tree = RangerTreeClassifier(regularization_factor=r)
            with pytest.raises(ValueError):
                tree.fit(iris_X, iris_y)

        # vector of ones isn't applied
        tree = RangerTreeClassifier(regularization_factor=[1] * iris_X.shape[1])
        tree.fit(iris_X, iris_y)
        assert tree.regularization_factor_ == []
        assert not tree.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        tree = RangerTreeClassifier(regularization_factor=reg)
        tree.fit(iris_X, iris_y)
        assert tree.regularization_factor_ == reg
        assert tree.use_regularization_factor_

    def test_always_split_features(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y, always_split_features=[0])
        # feature 0 is in every tree split
        for tree in tree.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_accuracy(self, iris_X, iris_y):
        X_train, X_test, y_train, y_test = train_test_split(
            iris_X, iris_y, test_size=0.33, random_state=42
        )

        # train and test a random forest classifier
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)

        # train and test a ranger classifier
        ra = RangerTreeClassifier()
        ra.fit(X_train, y_train)
        y_pred_ra = ra.predict(X_test)
        ranger_acc = accuracy_score(y_test, y_pred_ra)

        # the accuracy should be good
        assert rf_acc > 0.9
        assert ranger_acc > 0.9

    def test_check_estimator(self):
        check_estimator(RangerTreeClassifier())

    def test_get_depth(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth > 0

    def test_get_n_leaves(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        leaves = tree.get_n_leaves()
        assert isinstance(leaves, int)
        assert np.all(leaves > 0)

    def test_apply(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        leaves = tree.apply(iris_X)
        assert isinstance(leaves, np.ndarray)
        assert np.all(leaves > 0)
        assert len(leaves) == len(iris_X)

    def test_decision_path(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
        paths = tree.decision_path(iris_X)
        assert isinstance(paths, csr_matrix)
        assert paths.shape[0] == len(iris_X)

    def test_tree_interface(self, iris_X, iris_y):
        tree = RangerTreeClassifier()
        tree.fit(iris_X, iris_y)
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
