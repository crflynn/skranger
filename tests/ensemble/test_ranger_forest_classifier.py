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
from sklearn.utils.validation import check_is_fitted

from skranger.ensemble import RangerForestClassifier


class TestRangerForestClassifier:
    def test_init(self):
        _ = RangerForestClassifier()

    def test_fit(self, iris_X, iris_y):
        rfc = RangerForestClassifier()
        with pytest.raises(NotFittedError):
            check_is_fitted(rfc)
        rfc.fit(iris_X, iris_y)
        check_is_fitted(rfc)
        assert hasattr(rfc, "classes_")
        assert hasattr(rfc, "n_classes_")
        assert hasattr(rfc, "ranger_forest_")
        assert hasattr(rfc, "ranger_class_order_")
        assert hasattr(rfc, "n_features_")

    def test_predict(self, iris_X, iris_y):
        rfc = RangerForestClassifier()
        rfc.fit(iris_X, iris_y)
        pred = rfc.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = rfc.predict(iris_X_record)
        assert len(pred) == 1

    def test_predict_proba(self, iris_X, iris_y):
        rfc = RangerForestClassifier()
        rfc.fit(iris_X, iris_y)
        pred = rfc.predict_proba(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = rfc.predict_proba(iris_X_record)
        assert len(pred) == 1

    def test_predict_log_proba(self, iris_X, iris_y):
        rfc = RangerForestClassifier()
        rfc.fit(iris_X, iris_y)
        pred = rfc.predict_log_proba(iris_X)
        assert len(pred) == iris_X.shape[0]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = rfc.predict_log_proba(iris_X_record)
        assert len(pred) == 1

    def test_serialize(self, iris_X, iris_y):
        tf = tempfile.TemporaryFile()
        rfc = RangerForestClassifier()
        rfc.fit(iris_X, iris_y)
        pickle.dump(rfc, tf)
        tf.seek(0)
        new_rfc = pickle.load(tf)
        pred = new_rfc.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_clone(self, iris_X, iris_y):
        rfc = RangerForestClassifier()
        rfc.fit(iris_X, iris_y)
        clone(rfc)

    def test_verbose(self, iris_X, iris_y, verbose, capfd):
        rfc = RangerForestClassifier(verbose=verbose)
        rfc.fit(iris_X, iris_y)
        captured = capfd.readouterr()
        if verbose:
            assert len(captured.out) > 0
        else:
            assert len(captured.out) == 0

    def test_importance(self, iris_X, iris_y, importance, scale_permutation_importance, local_importance):
        rfc = RangerForestClassifier(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)
            return

        rfc.fit(iris_X, iris_y)
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

    def test_importance_pvalues(self, iris_X, iris_y, importance):
        rfc = RangerForestClassifier(
            importance=importance,
            scale_permutation_importance=scale_permutation_importance,
            local_importance=local_importance,
        )

        if importance not in ["none", "impurity", "impurity_corrected", "permutation"]:
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)
            return

        if not importance == "impurity_corrected"
            with pytest.raises(Exception):
                rfc.fit(iris_X, iris_y)
            return

        rfc.fit(iris_X, iris_y)
        assert len(rxf.get_importance_pvalues()) == iris_X.shape[1]

    def test_mtry(self, iris_X, iris_y, mtry):
        rfc = RangerForestClassifier(mtry=mtry)

        if callable(mtry) and mtry(5) > 5:
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)
            return
        elif not callable(mtry) and (mtry < 0 or mtry > iris_X.shape[0]):
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)
            return

        rfc.fit(iris_X, iris_y)
        if callable(mtry):
            assert rfc.mtry_ == mtry(iris_X.shape[1])
        else:
            assert rfc.mtry_ == mtry

    def test_inbag(self, iris_X, iris_y):
        inbag = [[1, 2, 3], [2, 3, 4]]
        rfc = RangerForestClassifier(n_estimators=2, inbag=inbag)
        rfc.fit(iris_X, iris_y)

        # inbag list different length from n_estimators
        rfc = RangerForestClassifier(n_estimators=1, inbag=inbag)
        with pytest.raises(ValueError):
            rfc.fit(iris_X, iris_y)

        # can't use inbag with sample weight
        rfc = RangerForestClassifier(inbag=inbag)
        with pytest.raises(ValueError):
            rfc.fit(iris_X, iris_y, sample_weight=[1] * len(iris_y))

        # can't use class sampling and inbag
        rfc = RangerForestClassifier(inbag=inbag, sample_fraction=[1, 1])
        with pytest.raises(ValueError):
            rfc.fit(iris_X, iris_y)

    def test_sample_fraction(self, iris_X, iris_y):
        rfc = RangerForestClassifier(sample_fraction=[0.69])
        rfc.fit(iris_X, iris_y)
        assert rfc.sample_fraction_ == [0.69]
        rfc = RangerForestClassifier(sample_fraction=0.69)
        rfc.fit(iris_X, iris_y)
        assert rfc.sample_fraction_ == [0.69]

        # test with single record
        iris_X_record = iris_X[0:1, :]
        pred = rfc.predict(iris_X_record)
        assert len(pred) == 1
        pred = rfc.predict_proba(iris_X_record)
        assert len(pred) == 1
        pred = rfc.predict_log_proba(iris_X_record)
        assert len(pred) == 1

    def test_sample_fraction_replace(self, iris_X, iris_y, replace):
        rfc = RangerForestClassifier(replace=replace)
        rfc.fit(iris_X, iris_y)

        if replace:
            assert rfc.sample_fraction_ == [1.0]
        else:
            assert rfc.sample_fraction_ == [0.632]

    def test_categorical_features(self, iris_X, iris_y, respect_categorical_features):
        # add a categorical feature
        categorical_col = np.atleast_2d(np.array([random.choice([0, 1]) for _ in range(iris_X.shape[0])]))
        iris_X_c = np.hstack((iris_X, categorical_col.transpose()))
        categorical_features = [iris_X.shape[1]]

        rfc = RangerForestClassifier(
            respect_categorical_features=respect_categorical_features, categorical_features=categorical_features
        )

        if respect_categorical_features not in ["partition", "ignore", "order"]:
            with pytest.raises(ValueError):
                rfc.fit(iris_X_c, iris_y)
            return

        rfc.fit(iris_X_c, iris_y)

        if respect_categorical_features in ("ignore", "order"):
            assert rfc.categorical_features_ == []
        else:
            assert rfc.categorical_features_ == [str(c).encode() for c in categorical_features]

    def test_split_rule(self, iris_X, iris_y, split_rule):
        rfc = RangerForestClassifier(split_rule=split_rule)

        if split_rule not in ["gini", "extratrees", "hellinger"]:
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)
            return

        # hellinger can only be used in binary classification
        if split_rule == "hellinger":
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)

        iris_2 = [0 if v == 2 else v for v in iris_y]
        rfc.fit(iris_X, iris_2)

        if split_rule == "gini":
            assert rfc.split_rule_ == 1
        elif split_rule == "extratrees":
            assert rfc.split_rule_ == 5
        if split_rule == "hellinger":
            assert rfc.split_rule_ == 7

        if split_rule == "extratrees":
            rfc = RangerForestClassifier(
                split_rule=split_rule, respect_categorical_features="partition", save_memory=True
            )
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)
        else:
            rfc = RangerForestClassifier(split_rule=split_rule, num_random_splits=2)
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)

    def test_regularization(self, iris_X, iris_y):
        rfc = RangerForestClassifier()
        rfc.fit(iris_X, iris_y)
        assert rfc.regularization_factor_ == []
        assert not rfc.use_regularization_factor_

        # vector must be between 0 and 1 and length matching feature num
        for r in [[1.1], [-0.1], [1, 1]]:
            rfc = RangerForestClassifier(regularization_factor=r)
            with pytest.raises(ValueError):
                rfc.fit(iris_X, iris_y)

        # vector of ones isn't applied
        rfc = RangerForestClassifier(regularization_factor=[1] * iris_X.shape[1])
        rfc.fit(iris_X, iris_y)
        assert rfc.regularization_factor_ == []
        assert not rfc.use_regularization_factor_

        # regularization vector is used
        reg = [0.5]
        rfc = RangerForestClassifier(regularization_factor=reg, n_jobs=2)
        # warns if n_jobs is not one since parallelization can't be used
        with pytest.warns(Warning):
            rfc.fit(iris_X, iris_y)
        assert rfc.n_jobs_ == 1
        assert rfc.regularization_factor_ == reg
        assert rfc.use_regularization_factor_

    def test_always_split_features(self, iris_X, iris_y):
        rfc = RangerForestClassifier(always_split_features=[0])
        rfc.fit(iris_X, iris_y)
        # feature 0 is in every tree split
        for tree in rfc.ranger_forest_["forest"]["split_var_ids"]:
            assert 0 in tree

    def test_accuracy(self, iris_X, iris_y):
        X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

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
