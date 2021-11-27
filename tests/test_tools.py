import pytest

from skranger.ensemble import RangerForestClassifier
from skranger.ensemble import RangerForestRegressor


@pytest.mark.skip()
def test_plot():
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_boston
    from sklearn.tree import plot_tree

    boston_X, boston_y = load_boston(return_X_y=True)
    forest = RangerForestRegressor(enable_tree_details=True)
    forest.fit(boston_X, boston_y)
    estimator = forest.get_estimator(0)
    plt.figure()
    plot_tree(
        estimator, impurity=False,  # impurity not yet implemented
    )
    plt.savefig(
        "tree.svg", bbox_inches="tight",  # don't truncate
    )


# FIXME not working yet
@pytest.mark.skip()
def test_shap_regressor(boston_X, boston_y):
    from shap import TreeExplainer

    forest = RangerForestRegressor()
    forest.fit(boston_X, boston_y)

    explainer = TreeExplainer(model=forest)
    shap_values = explainer.shap_values(boston_X)
    print(shap_values)


# FIXME not working yet
@pytest.mark.skip()
def test_shap_classifier(iris_X, iris_y):
    from shap import TreeExplainer

    forest = RangerForestClassifier(enable_tree_details=True)
    forest.fit(iris_X, iris_y)

    explainer = TreeExplainer(model=forest)
    shap_values = explainer.shap_values(iris_X)
    print(shap_values)


@pytest.mark.skip()
def test_shap_sklearn_regressor(boston_X, boston_y):
    from shap import TreeExplainer
    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor()
    forest.fit(boston_X, boston_y)

    explainer = TreeExplainer(model=forest)
    shap_values = explainer.shap_values(boston_X)
    print(shap_values)


@pytest.mark.skip()
def test_shap_sklearn_classifier(iris_X, iris_y):
    from shap import TreeExplainer
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier()
    forest.fit(iris_X, iris_y)

    explainer = TreeExplainer(model=forest)
    shap_values = explainer.shap_values(iris_X)
    print(shap_values)
