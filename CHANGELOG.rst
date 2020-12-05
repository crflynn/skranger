Release Changelog
-----------------

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
