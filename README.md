# skranger

`skranger` is a python package which provides [scikit-learn](https://scikit-learn.org/stable/index.html) compatible bindings to the C++ random forest implementation, [ranger](https://github.com/imbs-hl/ranger), using [Cython](https://cython.readthedocs.io/en/latest/).

There are two `sklearn` compatible classes, `RangerForestClassifier` and `RangerForestRegressor`. There is also the `RangerForestSurvival` class, which aims to be compatible with the [scikit-survival](https://github.com/sebp/scikit-survival) API,