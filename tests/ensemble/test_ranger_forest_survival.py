import pickle
import tempfile

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
