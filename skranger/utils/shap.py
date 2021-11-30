from contextlib import contextmanager

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree._tree import Tree as SKTree

from skranger.ensemble import RangerForestClassifier
from skranger.ensemble import RangerForestRegressor
from skranger.tree._tree import Tree


def _getclass(klass):
    """Get a __getattribute__ patch to override __class__."""

    def __class__override(self, item):
        if item == "__class__":
            return klass
        return object.__getattribute__(self, item)

    return __class__override


@contextmanager
def shap_patch():
    """Trick shap into thinking skranger objects are sklearn objects."""
    tree_orig = Tree.__getattribute__
    Tree.__getattribute__ = _getclass(SKTree)
    reg_orig = RangerForestRegressor.__getattribute__
    RangerForestRegressor.__getattribute__ = _getclass(RandomForestRegressor)
    cls_orig = RangerForestClassifier.__getattribute__
    RangerForestClassifier.__getattribute__ = _getclass(RandomForestClassifier)
    yield
    Tree.__getattribute__ = tree_orig
    RangerForestRegressor.__getattribute__ = reg_orig
    RangerForestClassifier.__getattribute__ = cls_orig
