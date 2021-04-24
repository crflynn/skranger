import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.io.arff import loadarff
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris


def load_veterans_lung_cancer():
    this_file = pathlib.Path(__file__)
    data_file = this_file.parent / "fixtures" / "veteran.arff"
    data = loadarff(data_file)
    df = pd.DataFrame(data=data[0], columns=list(data[1].names()))
    df["y"] = list(zip(df["Status"] == b"dead", df["Survival_in_days"]))
    y = df["y"]
    X = df.drop(["y", "Status", "Survival_in_days"], axis=1)
    return X, y


_boston_X, _boston_y = load_boston(return_X_y=True)
_iris_X, _iris_y = load_iris(return_X_y=True)
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


@pytest.fixture(params=["none", "random", "const"])
def mod(request):
    return request.param


@pytest.fixture
def iris_X_mod(mod):
    if mod == "none":
        return _iris_X
    elif mod == "random":
        np.random.seed(42)
        return np.concatenate((_iris_X, np.random.uniform(size=(_iris_X.shape))), 1)
    elif mod == "const":
        np.random.seed(42)
        return np.concatenate((_iris_X, np.random.uniform(size=(_iris_X.shape)), np.zeros(shape=(_iris_X.shape))), 1)


@pytest.fixture
def boston_X_mod(mod):
    if mod == "none":
        return _boston_X
    elif mod == "random":
        np.random.seed(42)
        return np.concatenate((_boston_X, np.random.uniform(size=(_boston_X.shape))), 1)
    elif mod == "const":
        np.random.seed(42)
        return np.concatenate(
            (_boston_X, np.random.uniform(size=(_boston_X.shape)), np.zeros(shape=(_boston_X.shape))), 1
        )


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


@pytest.fixture(
    params=["none", "impurity", "impurity_corrected", "permutation", "INVALID"]
)
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
    params=[
        "extratrees",
        "gini",
        "hellinger",
        "variance",
        "maxstat",
        "beta",
        "C",
        "C_ignore_ties",
        "INVALID",
    ]
)
def split_rule(request):
    return request.param
