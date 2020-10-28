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
    # [18.39205325 21.41698333 14.29509221 35.34981667 27.64378333 20.98569135
    #  21.15996673 14.0288093   9.44657947 29.99185    19.3774     11.88189465
    #  ...
    #  11.08502822 36.80993636 18.29633154 12.90448354 20.94311667 11.45154934
    #  41.44466667]

    # enable quantile regression on instantiation
    rfr = RangerForestRegressor(quantiles=True)
    rfr.fit(X_train, y_train)

    quantile_lower = rfr.predict_quantiles(X_test, quantiles=[0.1])
    print(quantile_lower)
    # [12.9 17.   8.  28.  22.  10.9  7.   8.   5.  20.8 16.9  7.   8.  18.
    #  22.  19.  29.  21.  19.  19.  22.  10.9 20.  16.  14.  20.   9.8 22.9
    #  ...
    #  16.  17.  12.  20.  13.  26.  19.  21.9  7.  14.9 13.   8.  17.9  7.9
    #  29. ]
    quantile_upper = rfr.predict_quantiles(X_test, quantiles=[0.9])
    print(quantile_upper)
    # [23.  27.  21.  44.  32.1 50.  50.  18.2 12.  43.  22.  17.  17.  24.
    #  31.1 25.  37.  28.  23.  24.  28.  18.  28.  23.  23.  26.  17.1 43.
    #  ...
    #  22.  24.  20.  28.  18.  44.2 24.  33.4 15.1 50.  21.  17.  25.  13.
    #  50. ]


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
