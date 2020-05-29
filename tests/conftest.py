import pytest
from sklearn.datasets import load_iris
from sksurv.datasets import load_veterans_lung_cancer

_iris_X, _iris_y = load_iris(True)
_lung_X, _lung_y = load_veterans_lung_cancer()


@pytest.fixture
def iris_X():
    return _iris_X


@pytest.fixture
def iris_y():
    return _iris_y


@pytest.fixture
def lung_X():
    # select only the numeric cols
    return _lung_X[["Age_in_years", "Karnofsky_score", "Months_from_Diagnosis"]]


@pytest.fixture
def lung_y():
    return _lung_y
