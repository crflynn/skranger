Release Changelog
-----------------

0.8.0rc1 (2022-04-25)
~~~~~~~~~~~~~~~~~~~~~

* Drop support for Python 3.7
* Add support for Windows
* Update scikit-learn constraint to 1.0

0.7.0 (2021-12-08)
~~~~~~~~~~~~~~~~~~

* Drop support for Python 3.6
* Fix quantile regression predictions for single record
* Fix Tree.value for classifiers
* Fix Tree.feature to use proper value for leaf nodes
* Allow ``skranger`` predictors to work with shap using ``skranger.utils.shap.shap_patch`` context manager
* Fix package includes to prevent installing extra files to ``site-packages``

0.6.1 (2021-09-05)
~~~~~~~~~~~~~~~~~~

* Use oldest supported numpy for builds

0.6.0 (2021-08-23)
~~~~~~~~~~~~~~~~~~

* Remove numpy from dependency spec; numpy is already a requirement of scikit-learn
* Change tree detail training code to be optional in ensembles due to expensive operations
* Change quantile regression to use ``np.quantile`` in lieu of `np.nanquantile`` for faster predictions
* Fix bug in tree classmethods when setting ``sample_fraction``
* Added more documentation around tree detail calculations

0.5.0 (2021-07-20)
~~~~~~~~~~~~~~~~~~

* Move ``split_select_weights``, ``always_split_features``, ``categorical_features`` params from init to fit methods.
* Sample weight checking is now a base class method.
* Remove sparse matrix args from bindings.
* Fix a bug with the output of ``predict_quantiles`` not being oriented properly for multiple quantiles
* Regression's ``predict_quantiles`` now requires a passed list of quantiles and the default is removed
* Regression's ``predict`` now takes an optional list of quantiles
* Remove ``snp_data`` and ``order_snps`` from bindings
* Moves ``class_weights`` to fit in classifier, and changes the arg type to a dictionary.
* Add ``RangerTreeClassifier``, ``RangerTreeRegressor``, and ``RangerTreeSurvival`` decision tree estimators which inherit between ``RangerMixin`` and ``BaseRangerTree``. Also provide a ``BaseRangerForest`` class for ensemble estimators.
* Add a low level ``Tree`` class which implements most of the ``sklearn.tree._tree.Tree`` interface.
* Fix incorrect documentation for ``num_random_splits``.

0.4.1 (2021-07-04)
~~~~~~~~~~~~~~~~~~

* Set an explicit lower bound on sklearn version.
* Fix a bug with ``split_select_weights``

0.4.0 (2021-04-23)
~~~~~~~~~~~~~~~~~~

* Add ``get_importance_pvalues`` method to estimators (thanks `kmacdon <https://github.com/kmacdon>`__)
* Add ``feature_importances_`` attribute, similar to sklearn forests
* Ensure ``self.respect_categorical_features`` is unchanged when fitting by introducing ``self.respect_categorical_features_``
* Change ``self.n_features_`` to ``self.n_features_in_``
* Add validation to classification targets, ensuring regression targets can't be passed to classifier
* Add sample weight validation to ensure that passing weights of ones results in identical output when passing None. We do this because ranger does additional RNG on weighted sampling when non-null weights are passed.
* Use ``self._validate_data`` in lieu of ``check_X_y`` when possible
* Use ``self._check_n_features`` in lieu of manually setting n features
* Add tags to estimators

0.3.2 (2021-01-18)
~~~~~~~~~~~~~~~~~~

* Fixed a bug related to incorrect ``sample_fraction`` input type
* Fixed a bug in which ``sample_fraction`` was being passed on predict, raising a ranger error

0.3.1 (2020-12-05)
~~~~~~~~~~~~~~~~~~

* Fixed a bug with incorrect output for quantile regression

0.3.0 (2020-10-28)
~~~~~~~~~~~~~~~~~~

* Enable quantile regression on RangerForestRegressor.

0.2.0 (2020-10-23)
~~~~~~~~~~~~~~~~~~

* Fix bug in classifier to reorder probabilities properly using forest's ``class_values``.

0.1.1 (2020-07-09)
~~~~~~~~~~~~~~~~~~

* Unpin sklearn/numpy deps, drop Cython dependency, since building wheels.
* Use numpy pyobjects to improve compilation.

0.1.0 (2020-06-03)
~~~~~~~~~~~~~~~~~~

* First release.
