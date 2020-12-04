skranger
========

|actions| |travis| |rtd| |pypi| |pyversions|

.. |actions| image:: https://github.com/crflynn/skranger/workflows/build/badge.svg
    :target: https://github.com/crflynn/skranger/actions

.. |travis| image:: https://img.shields.io/travis/crflynn/skranger-wheels/master.svg?logo=travis&label=wheels
    :target: https://travis-ci.org/crflynn/skranger-wheels

.. |rtd| image:: https://img.shields.io/readthedocs/skranger.svg
    :target: http://skranger.readthedocs.io/en/latest/

.. |pypi| image:: https://img.shields.io/pypi/v/skranger.svg
    :target: https://pypi.python.org/pypi/skranger

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/skranger.svg
    :target: https://pypi.python.org/pypi/skranger

``skranger`` provides `scikit-learn <https://scikit-learn.org/stable/index.html>`__ compatible Python bindings to the C++ random forest implementation, `ranger <https://github.com/imbs-hl/ranger>`__, using `Cython <https://cython.readthedocs.io/en/latest/>`__.

The latest release of ``skranger`` uses version `0.12.1 <https://github.com/imbs-hl/ranger/releases/tag/0.12.1>`__ of ``ranger``.


Installation
------------

``skranger`` is available on `pypi <https://pypi.org/project/skranger>`__ and can be installed via pip:

.. code-block:: bash

    pip install skranger


Usage
-----

There are two ``sklearn`` compatible classes, ``RangerForestClassifier`` and ``RangerForestRegressor``. There is also the ``RangerForestSurvival`` class, which aims to be compatible with the `scikit-survival <https://github.com/sebp/scikit-survival>`__ API.


RangerForestClassifier
~~~~~~~~~~~~~~~~~~~~~~

The ``RangerForestClassifier`` predictor uses ``ranger``'s ForestProbability class to enable both ``predict`` and ``predict_proba`` methods.

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from skranger.ensemble import RangerForestClassifier

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rfc = RangerForestClassifier()
    rfc.fit(X_train, y_train)

    predictions = rfc.predict(X_test)
    print(predictions)
    # [1 2 0 0 0 0 1 2 1 1 2 2 2 1 1 0 1 1 0 1 1 1 0 2 1 0 0 1 2 2 0 1 2 2 0 2 0 0]

    probabilities = rfc.predict_proba(X_test)
    print(probabilities)
    # [[0.01333333 0.98666667 0.        ]
    #  [0.         0.         1.        ]
    #  ...
    #  [0.98746032 0.01253968 0.        ]
    #  [0.99       0.01       0.        ]]


RangerForestRegressor
~~~~~~~~~~~~~~~~~~~~~

The ``RangerForestRegressor`` predictor uses ``ranger``'s ForestRegression class. It also supports quantile regression using the ``predict_quantiles`` method.

.. code-block:: python

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from skranger.ensemble import RangerForestRegressor

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rfr = RangerForestRegressor()
    rfr.fit(X_train, y_train)

    predictions = rfr.predict(X_test)
    print(predictions)
    # [26.27401667  8.96549989 24.82981667 27.92506667 28.04606667 45.4693
    #  21.89681787 40.30345    11.53959613 19.13675    15.88567273 16.69713567
    #  ...
    #  20.29025364 26.21245833 23.79643333 14.03546362 21.24893333 34.8825
    #  21.22463333]

    # enable quantile regression on instantiation
    rfr = RangerForestRegressor(quantiles=True)
    rfr.fit(X_train, y_train)

    quantile_lower = rfr.predict_quantiles(X_test, quantiles=[0.1])
    print(quantile_lower)
    # [22.    5.   21.88 23.08 23.1  35.89 10.85 31.5   7.04 14.5  11.7  10.9
    #   8.1  28.38  7.2  19.6  29.1  13.1  24.94 21.09 15.6  11.7  10.41 14.5
    #  ...
    #  18.9  21.4   9.43  8.7  26.46 18.99  7.2  19.27 18.5  21.19 18.99 18.88
    #  14.07 21.87 22.18  9.43 17.28 29.6  18.2 ]
    quantile_upper = rfr.predict_quantiles(X_test, quantiles=[0.9])
    print(quantile_upper)
    # [30.83 12.85 29.01 33.1  33.1  50.   29.75 50.   15.   23.   19.96 21.4
    #  20.53 50.   13.35 25.   48.5  19.6  46.   26.6  23.7  20.1  17.8  21.4
    #  ...
    #  26.78 28.1  17.86 27.5  46.25 24.4  16.74 24.4  28.7  29.1  24.4  25.
    #  25.   31.51 28.   20.8  26.7  42.13 24.24]


RangerForestSurvival
~~~~~~~~~~~~~~~~~~~~

The ``RangerForestSurvival`` predictor uses ``ranger``'s ForestSurvival class, and has an interface similar to the RandomSurvivalForest found in the ``scikit-survival`` package.

.. code-block:: python

    from sksurv.datasets import load_veterans_lung_cancer
    from sklearn.model_selection import train_test_split
    from skranger.ensemble import RangerForestSurvival

    X, y = load_veterans_lung_cancer()
    # select the numeric columns as features
    X = X[["Age_in_years", "Karnofsky_score", "Months_from_Diagnosis"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rfs = RangerForestSurvival()
    rfs.fit(X_train, y_train)

    predictions = rfs.predict(X_test)
    print(predictions)
    # [107.99634921  47.41235714  88.39933333  91.23566667  61.82104762
    #   61.15052381  90.29888492  47.88706349  21.25111508  85.5768254
    #   ...
    #   56.85498016  53.98227381  48.88464683  95.58649206  48.9142619
    #   57.68516667  71.96549206 101.79123016  58.95402381  98.36299206]

    chf = rfs.predict_cumulative_hazard_function(X_test)
    print(chf)
    # [[0.04233333 0.0605     0.24305556 ... 1.6216627  1.6216627  1.6216627 ]
    #  [0.00583333 0.00583333 0.00583333 ... 1.55410714 1.56410714 1.58410714]
    #  ...
    #  [0.12933333 0.14766667 0.14766667 ... 1.64342857 1.64342857 1.65342857]
    #  [0.00983333 0.0112619  0.04815079 ... 1.79304365 1.79304365 1.79304365]]

    survival = rfs.predict_survival_function(X_test)
    print(survival)
    # [[0.95855021 0.94129377 0.78422794 ... 0.19756993 0.19756993 0.19756993]
    #  [0.99418365 0.99418365 0.99418365 ... 0.21137803 0.20927478 0.20513086]
    #  ...
    #  [0.87868102 0.86271864 0.86271864 ... 0.19331611 0.19331611 0.19139258]
    #  [0.99021486 0.98880127 0.95299007 ... 0.16645277 0.16645277 0.16645277]]


License
-------

``skranger`` is licensed under `GPLv3 <https://github.com/crflynn/skranger/blob/master/LICENSE.txt>`__.

Development
-----------

To develop locally, it is recommended to have ``asdf``, ``make`` and a C++ compiler already installed. After cloning, run ``make setup``. This will setup the ranger submodule, install python and poetry from ``.tool-versions``, install dependencies using poetry, copy the ranger source code into skranger, and then build and install skranger in the local virtualenv.

To format code, run ``make fmt``. This will run isort and black against the .py files.

To run tests and inspect coverage, run ``make test``.

To rebuild in place after making changes, run ``make build``.

To create python package artifacts, run ``make dist``.

To build and view documentation, run ``make docs``.
