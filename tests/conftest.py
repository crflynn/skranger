import pytest
from sklearn.datasets import load_iris

_iris_X, _iris_y = load_iris(True)

@pytest.fixture
def iris_X():
    return _iris_X

@pytest.fixture
def iris_y():
    return _iris_y
