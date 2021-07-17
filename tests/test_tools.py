import pytest

from skranger.ensemble import RangerForestRegressor


@pytest.mark.skip()
def test_plot():
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_boston
    from sklearn.tree import plot_tree

    boston_X, boston_y = load_boston(return_X_y=True)
    forest = RangerForestRegressor()
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
def test_shap(boston_X, boston_y):
    from shap import TreeExplainer

    forest = RangerForestRegressor()
    forest.fit(boston_X, boston_y)

    explainer = TreeExplainer(model=forest, data=boston_X)
    shap_values = explainer.shap_values(boston_X, check_additivity=True)
    print(shap_values)


@pytest.mark.skip()
def test_shap_sklearn(boston_X, boston_y):
    from shap import TreeExplainer
    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor()
    forest.fit(boston_X, boston_y)

    explainer = TreeExplainer(model=forest, data=boston_X)
    shap_values = explainer.shap_values(boston_X, check_additivity=True)
    print(shap_values)
