import pickle
import tempfile

from sklearn.base import clone

from skranger.ensemble import RangerForestSurvival


class TestRangerForestSurvival:
    def test_init(self):
        _ = RangerForestSurvival()

    def test_fit(self, iris_X, iris_y):
        rfc = RangerForestSurvival()
        rfc.fit(iris_X, iris_y)
        assert hasattr(rfc, "classes_")
        assert hasattr(rfc, "n_classes_")
        assert hasattr(rfc, "ranger_forest_")
        assert hasattr(rfc, "n_features_")

    def test_predict(self, iris_X, iris_y):
        rfc = RangerForestSurvival()
        rfc.fit(iris_X, iris_y)
        pred = rfc.predict(iris_X)
        print(pred)
        assert len(pred) == iris_X.shape[0]

    def test_serialize(self, iris_X, iris_y):
        tf = tempfile.TemporaryFile()
        rfc = RangerForestSurvival()
        rfc.fit(iris_X, iris_y)
        pickle.dump(rfc, tf)
        tf.seek(0)
        new_rfc = pickle.load(tf)
        pred = new_rfc.predict(iris_X)
        assert len(pred) == iris_X.shape[0]

    def test_clone(self, iris_X, iris_y):
        rfc = RangerForestSurvival()
        rfc.fit(iris_X, iris_y)
        clone(rfc)
