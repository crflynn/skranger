Release Changelog
-----------------

0.4.1 (2021-06-29)
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
