from skranger import RandomForestClassifier


class TestRandomForestClassifier:

    def test_init(self):
        _ = RandomForestClassifier()

    def test_train(self, iris_X, iris_y):
        rfc = RandomForestClassifier()
        rfc.fit(iris_X, iris_y)
        print("predicting")
        pred = rfc.predict(iris_X)
        print(pred)
