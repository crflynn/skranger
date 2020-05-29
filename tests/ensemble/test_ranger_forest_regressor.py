import pickle
import tempfile

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
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
        assert hasattr(rfr, "n_features_")

    def test_predict(self, boston_X, boston_y):
        rfr = RangerForestRegressor()
        rfr.fit(boston_X, boston_y)
        pred = rfr.predict(boston_X)
        assert len(pred) == boston_X.shape[0]

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
