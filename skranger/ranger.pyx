import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport ranger_

# Enums required as input for the ForestClassifier
cpdef enum MemoryMode:
    MEM_DOUBLE = 0,
    MEM_FLOAT = 1,
    MEM_CHAR = 2

# the c++ code is mis-ordered also
cpdef enum ImportanceMode:
    IMP_NONE = 0,
    IMP_GINI = 1,
    IMP_PERM_BREIMAN = 2,
    IMP_PERM_LIAW = 4,
    IMP_PERM_RAW = 3,
    IMP_GINI_CORRECTED = 5,
    IMP_PERM_CASEWISE = 6

cpdef enum SplitRule:
    LOGRANK = 1,
    AUC = 2,
    AUC_IGNORE_TIES = 3,
    MAXSTAT = 4,
    EXTRATREES = 5,
    BETA = 6,
    HELLINGER = 7

cpdef enum PredictionType:
    RESPONSE = 1,
    TERMINALNODES = 2


cdef class DataNumpy:
    """Cython wrapper for DataNumpy class in C++.

    This wraps the Data class in C++, which encapsulates training data passed to the
    random forest classes. It allows us to pass numpy arrays into the Data object as
    pointers which get stored in vectors.
    """
    cdef unique_ptr[ranger_.DataNumpy] c_data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,
        np.ndarray[double, ndim=2, mode="fortran"] x not None,
        np.ndarray[double, ndim=1, mode="fortran"] y not None,
        vector[string] variable_names,
    ):
        cdef int num_rows, num_cols
        num_rows = x.shape[0]
        num_cols = x.shape[1]
        self.c_data.reset(
            new ranger_.DataNumpy(
                &x[0, 0],
                &y[0],
                variable_names,
                num_rows,
                num_cols
            )
        )

    def get_x(self, size_t row, size_t col):
        return dereference(self.c_data).get_x(row, col)

    def get_y(self, size_t row, size_t col):
        return dereference(self.c_data).get_y(row, col)

    def reserve_memory(self, size_t y_cols):
        return dereference(self.c_data).reserveMemory(y_cols)

    def set_x(self, size_t col, size_t row, double value, bool& error):
        return dereference(self.c_data).set_x(col, row, value, error)

    def set_y(self, size_t col, size_t row, double value, bool& error):
        return dereference(self.c_data).set_y(col, row, value, error)

cdef class ForestClassification:
    """Cython wrapper for ranger's ForestClassification class."""
    cdef unique_ptr[ranger_.ForestClassification] c_fc

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,
        ranger_.MemoryMode memory_mode,
        np.ndarray[double, ndim=2, mode="fortran"] x,
        np.ndarray[double, ndim=1, mode="fortran"] y,
        int mtry,
        char* output_prefix,
        int num_trees,
        int seed,
        int num_threads,
        ranger_.ImportanceMode importance_mode,
        int min_node_size,
        bool prediction_mode,
        bool sample_with_replacement,
        const vector[string]& unordered_variable_names,
        bool memory_saving_splitting,
        ranger_.SplitRule splitrule,
        bool predict_all,
        vector[double]& sample_fraction,
        double alpha,
        double minprop,
        bool holdout,
        ranger_.PredictionType prediction_type,
        int num_random_splits,
        bool order_snps,
        int max_depth,
        const vector[double]& regularization_factor,
        bool regularization_usedepth
    ):
        variable_names = [str(r).encode() for r in range(x.shape[1])]
        data = DataNumpy(x, y, variable_names)
        self.c_fc.reset(new ranger_.ForestClassification())
        dereference(self.c_fc).init(
            memory_mode,
            ranger_.move(data.c_data),
            mtry,
            output_prefix,
            num_trees,
            seed,
            num_threads,
            importance_mode,
            min_node_size,
            prediction_mode,
            sample_with_replacement,
            unordered_variable_names,
            memory_saving_splitting,
            splitrule,
            predict_all,
            sample_fraction,
            alpha,
            minprop,
            holdout,
            prediction_type,
            num_random_splits,
            order_snps,
            max_depth,
            regularization_factor,
            regularization_usedepth,
        )

    def run(self, bool verbose, bool compute_oob_error):
        dereference(self.c_fc).run(verbose, compute_oob_error)
