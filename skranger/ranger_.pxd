from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


# https://stackoverflow.com/questions/29571780/cython-avoid-copy-through-stdmove-not-working
# C++ move is needed for us to pass ``unique_ptr``s to C++ using Cython
cdef extern from * namespace "ranger":
    """
    namespace ranger {

    template <typename T>
    inline typename std::remove_reference<T>::type&& move(T& t) {
        return std::move(t);
    }

    template <typename T>
    inline typename std::remove_reference<T>::type&& move(T&& t) {
        return std::move(t);
    }

    }  // namespace ranger
    """
    cdef T move[T](T)

# Enums required as input for the ForestClassifier
cdef extern from "../ranger/cpp_version/src/globals.h" namespace "ranger":
    cpdef enum MemoryMode:
        MEM_DOUBLE = 0,
        MEM_FLOAT = 1,
        MEM_CHAR = 2

cdef extern from "../ranger/cpp_version/src/globals.h" namespace "ranger":
    # the c++ code is mis-ordered also
    cpdef enum ImportanceMode:
        IMP_NONE = 0,
        IMP_GINI = 1,
        IMP_PERM_BREIMAN = 2,
        IMP_PERM_LIAW = 4,
        IMP_PERM_RAW = 3,
        IMP_GINI_CORRECTED = 5,
        IMP_PERM_CASEWISE = 6

cdef extern from "../ranger/cpp_version/src/globals.h" namespace "ranger":
    cpdef enum SplitRule:
        LOGRANK = 1,
        AUC = 2,
        AUC_IGNORE_TIES = 3,
        MAXSTAT = 4,
        EXTRATREES = 5,
        BETA = 6,
        HELLINGER = 7

cdef extern from "../ranger/cpp_version/src/globals.h" namespace "ranger":
    cpdef enum PredictionType:
        RESPONSE = 1,
        TERMINALNODES = 2

# Inherit from the Data class so that we can pass numpy array based data objects
cdef extern from "../ranger/cpp_version/src/utility/Data.cpp":
    pass

cdef extern from "../ranger/cpp_version/src/utility/Data.h" namespace "ranger":
    cdef cppclass Data:
        Data() except +
        double get_x(size_t row, size_t col)
        double get_y(size_t row, size_t col)
        size_t getVariableID(char* variable_name)
        void reserveMemory(size_t y_cols)
        void set_x(size_t col, size_t row, double value, bool& error)
        void set_y(size_t col, size_t row, double value, bool& error)

        size_t getSnp(size_t row, size_t col, size_t col_permuted)
        size_t getPermutedSampleID(size_t sampleID)
        size_t getUnpermutedVarID(size_t varID)

# Custom Data child class for numpy
cdef extern from "DataNumpy.h" namespace "ranger":
    cdef cppclass DataNumpy(Data):
        DataNumpy() except +
        DataNumpy(
            double* x,
            double* y,
            vector[string] variable_names,
            size_t num_rows,
            size_t num_cols
        )

# Inherit from the forest class for the classifier
cdef extern from "../ranger/cpp_version/src/Forest/Forest.cpp":
    pass


cdef extern from "../ranger/cpp_version/src/Forest/Forest.h" namespace "ranger":
    cdef cppclass Forest:
        Forest() except +
        void init(
            MemoryMode memory_mode,
            unique_ptr[DataNumpy] input_data,
            int mtry,
            char* output_prefix,
            int num_trees,
            int seed,
            int num_threads,
            int importance_mode,
            int min_node_size,
            bool prediction_mode,
            bool sample_with_replacement,
            const vector[string]& unordered_variable_names,
            bool memory_saving_splitting,
            SplitRule splitrule,
            bool predict_all,
            vector[double]& sample_fraction,
            double alpha,
            double minprop,
            bool holdout,
            PredictionType prediction_type,
            int num_random_splits,
            bool order_snps,
            int max_depth,
            const vector[double]& regularization_factor,
            bool regularization_usedepth
        )
        void run(bool verbose, bool compute_oob_error)

# Needed reference to be able to import in python
cdef extern from "../ranger/cpp_version/src/utility/utility.cpp":
    pass

cdef extern from "../ranger/cpp_version/src/utility/utility.h" namespace "ranger":
    void equalSplit(
        vector[unsigned int]& result,
        unsigned int start,
        unsigned int end,
        unsigned int num_parts
    )

# Needed reference to be able to import in python
cdef extern from "../ranger/cpp_version/src/Tree/Tree.cpp":
    pass

cdef extern from "../ranger/cpp_version/src/Tree/Tree.h" namespace "ranger":
    cdef cppclass Tree:
        Tree() except +
        Tree(
            vector[size_t]& child_nodeIDs,
            vector[size_t]& split_varIDs,
            vector[double]& split_values
        )

# Needed reference to be able to import in python
cdef extern from "../ranger/cpp_version/src/Tree/TreeClassification.cpp":
    pass

cdef extern from "../ranger/cpp_version/src/Tree/TreeClassification.h" namespace "ranger":
    cdef cppclass TreeClassification(Tree):
        TreeClassification() except +
        TreeClassification(
            vector[double]* class_values,
            vector[unsigned int]* response_classIDs,
            vector[vector[size_t]]* sampleIDs_per_class,
            vector[double]* class_weights
        )

# The classifier, inherits from Forest
cdef extern from "../ranger/cpp_version/src/Forest/ForestClassification.cpp":
    pass

cdef extern from "../ranger/cpp_version/src/Forest/ForestClassification.h" namespace "ranger":
    cdef cppclass ForestClassification(Forest):
        ForestClassification() except +
