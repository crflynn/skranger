"""Cython definition file for C++ classes from ranger, skranger, and C++ std."""
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


# Wrap stringstream for when verbose == False
cdef extern from "<sstream>" namespace "std":
    cdef cppclass stringstream:
        stringstream() except +

# Wrap cout/iostream for when verbose == True
# https://stackoverflow.com/questions/30984078/cython-working-with-c-streams
cdef extern from "<iostream>" namespace "std":
    ostream cout
    cdef cppclass ostream:
        ostream& write(const char*, int) except +

# Enums required as input for the ForestClassifier
cdef extern from "./ranger/src/globals.h" namespace "ranger":
    cpdef enum MemoryMode:
        MEM_DOUBLE = 0,
        MEM_FLOAT = 1,
        MEM_CHAR = 2

cdef extern from "./ranger/src/globals.h" namespace "ranger":
    # the c++ code is mis-ordered also
    cpdef enum ImportanceMode:
        IMP_NONE = 0,
        IMP_GINI = 1,
        IMP_PERM_BREIMAN = 2,
        IMP_PERM_LIAW = 4,
        IMP_PERM_RAW = 3,
        IMP_GINI_CORRECTED = 5,
        IMP_PERM_CASEWISE = 6

cdef extern from "./ranger/src/globals.h" namespace "ranger":
    cpdef enum SplitRule:
        LOGRANK = 1,
        AUC = 2,
        AUC_IGNORE_TIES = 3,
        MAXSTAT = 4,
        EXTRATREES = 5,
        BETA = 6,
        HELLINGER = 7

cdef extern from "./ranger/src/globals.h" namespace "ranger":
    cpdef enum PredictionType:
        RESPONSE = 1,
        TERMINALNODES = 2

cdef extern from "./ranger/src/globals.h" namespace "ranger":
    cpdef enum TreeType:
        TREE_CLASSIFICATION = 1,
        TREE_REGRESSION = 3,
        TREE_SURVIVAL = 5,
        TREE_PROBABILITY = 9

# Inherit from the Data class so that we can pass numpy array based data objects
cdef extern from "./ranger/src/utility/Data.cpp":
    pass

cdef extern from "./ranger/src/utility/Data.h" namespace "ranger":
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
            size_t num_cols,
            size_t num_cols_y
        )

cdef extern from "./ranger/src/Tree/Tree.cpp":
    pass

cdef extern from "./ranger/src/Tree/Tree.h" namespace "ranger":
    cdef cppclass Tree:
        Tree() except +
        Tree(
            vector[vector[size_t]]& child_nodeIDs,
            vector[size_t]& split_varIDs,
            vector[double]& split_values
        )

cdef extern from "./ranger/src/Tree/TreeClassification.cpp":
    pass

cdef extern from "./ranger/src/Tree/TreeClassification.h" namespace "ranger":
    cdef cppclass TreeClassification(Tree):
        TreeClassification() except +
        TreeClassification(
            vector[double]* class_values,
            vector[unsigned int]* response_classIDs,
            vector[vector[size_t]]* sampleIDs_per_class,
            vector[double]* class_weights
        )

cdef extern from "./ranger/src/Tree/TreeSurvival.cpp":
    pass

cdef extern from "./ranger/src/Tree/TreeSurvival.h" namespace "ranger":
    cdef cppclass TreeSurvival(Tree):
        TreeSurvival() except +
        TreeSurvival(
            vector[double]* unique_timepoints,
            vector[size_t]* response_timepointIDs,
        )

cdef extern from "./ranger/src/Tree/TreeRegression.cpp":
    pass

cdef extern from "./ranger/src/Tree/TreeRegression.h" namespace "ranger":
    cdef cppclass TreeRegression(Tree):
        TreeRegression() except +
        TreeRegression(
            vector[vector[size_t]]& child_nodeIDs,
            vector[size_t]& split_varIDs,
        )

cdef extern from "./ranger/src/Tree/TreeProbability.cpp":
    pass

cdef extern from "./ranger/src/Tree/TreeProbability.h" namespace "ranger":
    cdef cppclass TreeProbability(Tree):
        TreeProbability() except +
        TreeProbability(
            vector[double]* class_values,
            vector[unsigned int]* response_classIDs,
            vector[vector[size_t]]* sampleIDs_per_class,
            vector[double]* class_weights
        )

cdef extern from "./ranger/src/Forest/Forest.cpp":
    pass

cdef extern from "./ranger/src/Forest/Forest.h" namespace "ranger":
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
        void initR(
            unique_ptr[DataNumpy] input_data,
            int mtry,
            int num_trees,
            ostream* verbose_out,
            int seed,
            int num_threads,
            int importance_mode,
            int min_node_size,
            vector[vector[double]]& split_select_weights,
            const vector[string]& always_split_variable_names,
            bool prediction_mode,
            bool sample_with_replacement,
            const vector[string]& unordered_variable_names,
            bool memory_saving_splitting,
            SplitRule splitrule,
            vector[double]& case_weights,
            vector[vector[size_t]]& manual_inbag,
            bool predict_all,
            bool keep_inbag,
            vector[double]& sample_fraction,
            double alpha,
            double minprop,
            bool holdout,
            PredictionType prediction_type,
            int num_random_splits,
            bool order_snps,
            int max_depth,
            const vector[double]& regularization_factor,
            bool regularization_usedepth,
        )
        void run(bool verbose, bool compute_oob_error)
        void saveToFile()
        vector[vector[vector[size_t]]] getChildNodeIDs()
        const vector[bool]& getIsOrderedVariable()
        unsigned int getMinNodeSize()
        unsigned int getMtry()
        size_t getNumIndependentVariables()
        size_t getNumTrees()
        double getOverallPredictionError()
        const vector[vector[vector[double]]]& getPredictions()
        vector[vector[double]] getSplitValues()
        vector[vector[size_t]] getSplitVarIDs()
        vector[vector[size_t]] getInbagCounts()
        const vector[double]& getVariableImportance()
        const vector[double]& getVariableImportanceCasewise()

cdef extern from "./ranger/src/utility/utility.cpp":
    pass

cdef extern from "./ranger/src/utility/utility.h" namespace "ranger":
    void equalSplit(
        vector[unsigned int]& result,
        unsigned int start,
        unsigned int end,
        unsigned int num_parts
    )

cdef extern from "./ranger/src/Forest/ForestClassification.cpp":
    pass

cdef extern from "./ranger/src/Forest/ForestClassification.h" namespace "ranger":
    cdef cppclass ForestClassification(Forest):
        ForestClassification() except +
        const vector[double]& getClassValues()
        void loadForest(
            size_t num_trees,
            vector[vector[vector[size_t]]]& forest_child_nodeIDs,
            vector[vector[size_t]]& forest_split_varIDs,
            vector[vector[double]]& forest_split_values,
            vector[double]& class_Values,
            vector[bool]& is_ordered_variable,
        )
        void setClassWeights(vector[double]& class_weights)

cdef extern from "./ranger/src/Forest/ForestRegression.cpp":
    pass

cdef extern from "./ranger/src/Forest/ForestRegression.h" namespace "ranger":
    cdef cppclass ForestRegression(Forest):
        ForestRegression() except +
        void loadForest(
            size_t num_trees,
            vector[vector[vector[size_t]]]& forest_child_nodeIDs,
            vector[vector[size_t]]& forest_split_varIDs,
            vector[vector[double]]& forest_split_values,
            vector[bool]& is_ordered_variable,
        )

cdef extern from "./ranger/src/Forest/ForestProbability.cpp":
    pass

cdef extern from "./ranger/src/Forest/ForestProbability.h" namespace "ranger":
    cdef cppclass ForestProbability(Forest):
        ForestProbability() except +
        vector[double]& getClassValues()
        vector[vector[vector[double]]] getTerminalClassCounts()
        void setClassWeights(vector[double]& class_weights)
        void loadForest(
            size_t num_trees,
            vector[vector[vector[size_t]]]& forest_child_nodeIDs,
            vector[vector[size_t]]& forest_split_varIDs,
            vector[vector[double]]& forest_split_values,
            vector[double]& class_Values,
            vector[vector[vector[double]]]& forest_terminal_class_counts,
            vector[bool]& is_ordered_variable,
        )

cdef extern from "./ranger/src/Forest/ForestSurvival.cpp":
    pass

cdef extern from "./ranger/src/Forest/ForestSurvival.h" namespace "ranger":
    cdef cppclass ForestSurvival(Forest):
        ForestSurvival() except +
        vector[vector[vector[double]]] getChf()
        vector[double]& getUniqueTimepoints()
        void loadForest(
            size_t num_trees,
            vector[vector[vector[size_t]]]& forest_child_nodeIDs,
            vector[vector[size_t]]& forest_split_varIDs,
            vector[vector[double]]& forest_split_values,
            vector[vector[vector[double]]]& forest_chf,
            vector[double]& unique_timepoints,
            vector[bool]& is_ordered_variable,
        )
