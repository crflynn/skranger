import pickle
import tempfile

from sklearn.base import clone

from skranger.ensemble import RangerForestSurvival

N = 10


class TestRangerForestSurvival:
    def test_init(self):
        _ = RangerForestSurvival()

    def test_fit(self, lung_X, lung_y):
        rfs = RangerForestSurvival(num_trees=N)
        rfs.fit(lung_X, lung_y)
        assert hasattr(rfs, "event_times_")
        assert hasattr(rfs, "cumulative_hazard_function_")
        assert hasattr(rfs, "ranger_forest_")
        assert hasattr(rfs, "n_features_")

    def test_predict(self, lung_X, lung_y):
        rfs = RangerForestSurvival(num_trees=N)
        rfs.fit(lung_X, lung_y)
        pred = rfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_serialize(self, lung_X, lung_y):
        tf = tempfile.TemporaryFile()
        rfs = RangerForestSurvival(num_trees=N)
        rfs.fit(lung_X, lung_y)
        pickle.dump(rfs, tf)
        tf.seek(0)
        new_rfs = pickle.load(tf)
        pred = new_rfs.predict(lung_X)
        assert len(pred) == lung_X.shape[0]

    def test_clone(self, lung_X, lung_y):
        rfs = RangerForestSurvival(num_trees=N)
        rfs.fit(lung_X, lung_y)
        clone(rfs)
