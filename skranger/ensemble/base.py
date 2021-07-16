import bisect

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skranger.base import RangerMixin


class BaseRangerForest(RangerMixin):
    @property
    def feature_importances_(self):
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'feature_importances_'"
            ) from None
        try:
            return self.ranger_forest_["variable_importance"]
        except KeyError:
            raise ValueError(
                "importance must be set to something other than 'none'"
            ) from None

    def get_importance_pvalues(self):
        """Calculate p-values for variable importances.

        Uses the fast method from Janitza et al. (2016).
        """
        check_is_fitted(self)
        if self.importance != "impurity_corrected":
            raise ValueError(
                "p-values can only be calculated with importance parameter set to 'impurity_corrected'"
            )

        vimp = np.array(self.ranger_forest_["variable_importance"])
        m1 = vimp[vimp < 0]
        m2 = vimp[vimp == 0]

        if len(m1) == 0:
            raise ValueError(
                "No negative importance values found, cannot calculate p-values."
            )
        if len(m2) < 1:
            vimp_dist = np.concatenate((m1, -m1))
        else:
            vimp_dist = np.concatenate((m1, -m1, m2))

        vimp_dist.sort()
        result = []
        for i in range(len(vimp)):
            result.append(bisect.bisect_left(vimp_dist, vimp[i]))
        pval = 1 - np.array(result) / len(vimp_dist)
        return pval
