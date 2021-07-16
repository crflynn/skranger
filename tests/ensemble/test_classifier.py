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
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from skranger.ensemble import RangerForestClassifier
from skranger.tree import RangerTreeClassifier


class TestRangerForestClassifier:
    def test_init(self):
        _ = RangerForestClassifier()

    def test_fit(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        with pytest.raises(NotFittedError):
            check_is_fitted(forest)
        forest.fit(iris_X, iris_y)
        check_is_fitted(forest)
        assert hasattr(forest, "classes_")
        assert hasattr(forest, "n_classes_")
        assert hasattr(forest, "ranger_forest_")
        assert hasattr(forest, "ranger_class_order_")
        assert hasattr(forest, "n_features_in_")

    def test_predict(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y)
        pred = forest.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = forest.predict(iris_X_record)
        assert len(pred) == 1

    def test_predict_proba(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y)
        pred = forest.predict_proba(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = forest.predict_proba(iris_X_record)
        assert len(pred) == 1

    def test_predict_log_proba(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y)
        pred = forest.predict_log_proba(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = forest.predict_log_proba(iris_X_record)
        assert len(pred) == 1

    def test_serialize(self, iris_X, iris_y):
        tf = tempfile.TemporaryFile()
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y)
        pickle.dump(forest, tf)
        tf.seek(0)
        new_forest = pickle.load(tf)
        pred = new_forest.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_clone(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y)
        clone(forest)

    def test_verbose(self, iris_X, iris_y, verbose, capfd):
        forest = RangerForestClassifier(verbose=verbose)
        forest.fit(iris_X, iris_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(
        self, iris_X, iris_y, importance, scale_permutation_importance, local_importance
    ):
        forest = RangerForestClassifier(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
            return

        forest.fit(iris_X, iris_y)
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

    def test_importance_pvalues(self, iris_X_mod, iris_y, importance, mod):
        forest = RangerForestClassifier(importance=importance)
        np.random.seed(42)

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                forest.fit(iris_X_mod, iris_y)
            return

        if not importance == "impurity_corrected":
            forest.fit(iris_X_mod, iris_y)
            with pytest.raises(ValueError):
                forest.get_importance_pvalues()
            return

        # Test error for no non-negative importance values
        if mod == "none":
            forest.fit(iris_X_mod, iris_y)
            with pytest.raises(ValueError):
                forest.get_importance_pvalues()
            return

        forest.fit(iris_X_mod, iris_y)
        assert len(forest.get_importance_pvalues()) == iris_X_mod.shape[1]

    def test_mtry(self, iris_X, iris_y, mtry):
        forest = RangerForestClassifier(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > iris_X.shape[0]):
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
            return

        forest.fit(iris_X, iris_y)
        if callable(mtry):
            assert forest.mtry_ == mtry(iris_X.shape[1])
        else:
            assert forest.mtry_ == mtry

    def test_inbag(self, iris_X, iris_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        forest = RangerForestClassifier(n_estimators=2, inbag=inbag)
        forest.fit(iris_X, iris_y)

        # inbag list different length from n_estimators
        forest = RangerForestClassifier(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            forest.fit(iris_X, iris_y)

        # can't use inbag with sample weight
        forest = RangerForestClassifier(inbag=inbag)
        with pytest.raises(ValueError):
            forest.fit(iris_X, iris_y, sample_weight=[1] * len(iris_y))

        # can't use class sampling and inbag
        forest = RangerForestClassifier(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            forest.fit(iris_X, iris_y)

    def test_sample_fraction(self, iris_X, iris_y):
        forest = RangerForestClassifier(sample_fraction=[0.69])
        forest.fit(iris_X, iris_y)
        assert forest.sample_fraction_ == [0.69]
        forest = RangerForestClassifier(sample_fraction=0.69)
        forest.fit(iris_X, iris_y)
        assert forest.sample_fraction_ == [0.69]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = forest.predict(iris_X_record)
        assert len(pred) == 1
        pred = forest.predict_proba(iris_X_record)
        assert len(pred) == 1
        pred = forest.predict_log_proba(iris_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, iris_X, iris_y, replace):
        forest = RangerForestClassifier(replace=replace)
        forest.fit(iris_X, iris_y)

        if replace:
            assert forest.sample_fraction_ == [1.0]
        else:
            assert forest.sample_fraction_ == [0.632]

    def test_categorical_features(self, iris_X, iris_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(
            np.array([random.choice([0, 1]) for _ in range(iris_X.shape[0])])
        )
        iris_X_c = np.hstack((iris_X, categorical_col.transpose()))
        categorical_features = [iris_X.shape[1]]

        forest = RangerForestClassifier(
            respect_categorical_features=respect_categorical_features,
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                forest.fit(iris_X_c, iris_y, categorical_features=categorical_features)
            return

        forest.fit(iris_X_c, iris_y, categorical_features=categorical_features)
        forest.predict(iris_X_c)

    def test_split_rule(self, iris_X, iris_y, split_rule):
        forest = RangerForestClassifier(split_rule=split_rule)

        if split_rule not in ["gini", "extratrees", "hellinger"]:
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
            return

        # hellinger can only be used in binary classification
        if split_rule == "hellinger":
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)

        iris_2 = [0 if v == 2 else v for v in iris_y]
        forest.fit(iris_X, iris_2)

        if split_rule == "gini":
            assert forest.split_rule_ == 1
        elif split_rule == "extratrees":
            assert forest.split_rule_ == 5
        if split_rule == "hellinger":
            assert forest.split_rule_ == 7

        if split_rule == "extratrees":
            forest = RangerForestClassifier(
                split_rule=split_rule,
                respect_categorical_features="partition",
                save_memory=True,
            )
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
        else:
            forest = RangerForestClassifier(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)

    def test_split_select_weights(self, iris_X, iris_y):
        n_trees = 10
        weights = [0.1] * iris_X.shape[1]
        forest = RangerForestClassifier(n_estimators=n_trees)
        forest.fit(iris_X, iris_y, split_select_weights=weights)

        weights = [0.1] * (iris_X.shape[1] - 1)
        forest = RangerForestClassifier(n_estimators=n_trees)

        with pytest.raises(RuntimeError):
            forest.fit(iris_X, iris_y, split_select_weights=weights)

        weights = [[0.1] * (iris_X.shape[1])] * n_trees
        forest = RangerForestClassifier(n_estimators=n_trees)
        forest.fit(iris_X, iris_y, split_select_weights=weights)

        weights = [[0.1] * (iris_X.shape[1])] * (n_trees + 1)
        forest = RangerForestClassifier(n_estimators=n_trees)
        with pytest.raises(RuntimeError):
            forest.fit(iris_X, iris_y, split_select_weights=weights)

    def test_regularization(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y)
        assert forest.regularization_factor_ == []
        assert not forest.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            forest = RangerForestClassifier(regularization_factor=r)
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)

        # vector of ones isn't applied
        forest = RangerForestClassifier(regularization_factor=[1] * iris_X.shape[1])
        forest.fit(iris_X, iris_y)
        assert forest.regularization_factor_ == []
        assert not forest.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        forest = RangerForestClassifier(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            forest.fit(iris_X, iris_y)
        assert forest.n_jobs_ == 1
        assert forest.regularization_factor_ == reg
        assert forest.use_regularization_factor_

    def test_always_split_features(self, iris_X, iris_y):
        forest = RangerForestClassifier()
        forest.fit(iris_X, iris_y, always_split_features=[0])
        # feature 0 is in every tree split
        for tree in forest.ranger_forest_["forest"]["split_var_ids"]:
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
        ra = RangerForestClassifier()
        ra.fit(X_train, y_train)
        y_pred_ra = ra.predict(X_test)
        ranger_acc = accuracy_score(y_test, y_pred_ra)

        # the accuracy should be good
        assert rf_acc > 0.9
        assert ranger_acc > 0.9

    def test_feature_importances_(self, iris_X, iris_y, importance, local_importance):
        forest = RangerForestClassifier(
            importance=importance, local_importance=local_importance
        )
        with pytest.raises(AttributeError):
            _ = forest.feature_importances_

        if importance == "INVALID":
            with pytest.raises(ValueError):
                forest.fit(iris_X, iris_y)
            return

        forest.fit(iris_X, iris_y)
        if importance == "none":
            with pytest.raises(ValueError):
                _ = forest.feature_importances_
        else:
            assert len(forest.feature_importances_) == iris_X.shape[1]

    def test_estimators_(self, iris_X, iris_y):
        forest = RangerForestClassifier(n_estimators=10)
        with pytest.raises(AttributeError):
            _ = forest.estimators_
        forest.fit(iris_X, iris_y)
        estimators = forest.estimators_
        assert len(estimators) == 10
        assert isinstance(estimators[0], RangerTreeClassifier)
        check_is_fitted(estimators[0])

    def test_get_estimator(self, iris_X, iris_y):
        forest = RangerForestClassifier(n_estimators=10)
        with pytest.raises(NotFittedError):
            _ = forest.get_estimator(idx=0)
        forest.fit(iris_X, iris_y)
        forest.predict(iris_X)
        estimator = forest.get_estimator(0)
        check_is_fitted(estimator)
        estimator.predict(iris_X)
        assert isinstance(estimator, RangerTreeClassifier)
        with pytest.raises(IndexError):
            _ = forest.get_estimator(idx=20)

    def test_check_estimator(self):
        check_estimator(RangerForestClassifier())
