import math

import pytest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sksurv.datasets import load_veterans_lung_cancer

_boston_X, _boston_y = load_boston(True)
_iris_X, _iris_y = load_iris(True)
_lung_X, _lung_y = load_veterans_lung_cancer()


@pytest.fixture
def boston_X():
    return _boston_X


@pytest.fixture
def boston_y():
    return _boston_y


@pytest.fixture
def iris_X():
    return _iris_X


@pytest.fixture(params=['std','rand','rand_const'])
def mod(request):
    return request.param


@pytest.fixture
def iris_X_mod(mod):
    if mod == 'std':
        return _iris_X
    elif mod == 'rand':
        np.random.seed(42)
        return np.concatenate((_iris_X, np.random.uniform(size=(_iris_X.shape[0],1))),1)
    elif mod == 'rand_const':
        np.random.seed(42)
        return np.concatenate((_iris_X, np.random.uniform(size=(_iris_X.shape[0],3)), np.zeros(shape=(_iris_X.shape[0],3))),1)
        


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


@pytest.fixture(params=[True, False])
def verbose(request):
    return request.param


@pytest.fixture(params=["none", "impurity", "impurity_corrected", "permutation", "INVALID"])
def importance(request):
    return request.param


@pytest.fixture(params=[True, False])
def scale_permutation_importance(request):
    return request.param


@pytest.fixture(params=[True, False])
def local_importance(request):
    return request.param


@pytest.fixture(params=[True, False])
def replace(request):
    return request.param


@pytest.fixture(params=["partition", "ignore", "order", "INVALID"])
def respect_categorical_features(request):
    return request.param


def mtry_callable(num_features):
    return math.floor(math.sqrt(num_features))


def mtry_callable_invalid(num_features):
    return 999


@pytest.fixture(params=[1, mtry_callable, mtry_callable_invalid, -1, 999])
def mtry(request):
    return request.param


@pytest.fixture(
    params=["extratrees", "gini", "hellinger", "variance", "maxstat", "beta", "C", "C_ignore_ties", "INVALID"]
)
def split_rule(request):
    return request.param
