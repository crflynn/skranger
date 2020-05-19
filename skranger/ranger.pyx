import sys

import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_unique
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move

cimport ranger_

# # Enums required as input for the ForestClassifier
# cpdef enum MemoryMode:
#     MEM_DOUBLE = 0,
#     MEM_FLOAT = 1,
#     MEM_CHAR = 2
#
# # the c++ code is mis-ordered also
# cpdef enum ImportanceMode:
#     IMP_NONE = 0,
#     IMP_GINI = 1,
#     IMP_PERM_BREIMAN = 2,
#     IMP_PERM_LIAW = 4,
#     IMP_PERM_RAW = 3,
#     IMP_GINI_CORRECTED = 5,
#     IMP_PERM_CASEWISE = 6
#
# cpdef enum SplitRule:
#     LOGRANK = 1,
#     AUC = 2,
#     AUC_IGNORE_TIES = 3,
#     MAXSTAT = 4,
#     EXTRATREES = 5,
#     BETA = 6,
#     HELLINGER = 7
#
# cpdef enum PredictionType:
#     RESPONSE = 1,
#     TERMINALNODES = 2


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
#
# cdef class ForestClassification:
#     """Cython wrapper for ranger's ForestClassification class."""
#     cdef unique_ptr[ranger_.ForestClassification] c_fc
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     def __cinit__(self,
#         ranger_.MemoryMode memory_mode,
#         np.ndarray[double, ndim=2, mode="fortran"] x,
#         np.ndarray[double, ndim=1, mode="fortran"] y,
#         int mtry,
#         char* output_prefix,
#         int num_trees,
#         int seed,
#         int num_threads,
#         ranger_.ImportanceMode importance_mode,
#         int min_node_size,
#         bool prediction_mode,
#         bool sample_with_replacement,
#         const vector[string]& unordered_variable_names,
#         bool memory_saving_splitting,
#         ranger_.SplitRule splitrule,
#         bool predict_all,
#         vector[double]& sample_fraction,
#         double alpha,
#         double minprop,
#         bool holdout,
#         ranger_.PredictionType prediction_type,
#         int num_random_splits,
#         bool order_snps,
#         int max_depth,
#         const vector[double]& regularization_factor,
#         bool regularization_usedepth
#     ):
#         variable_names = [str(r).encode() for r in range(x.shape[1])]
#         data = DataNumpy(x, y, variable_names)
#         self.c_fc.reset(new ranger_.ForestClassification())
#         dereference(self.c_fc).init(
#             memory_mode,
#             move(data.c_data),
#             mtry,
#             output_prefix,
#             num_trees,
#             seed,
#             num_threads,
#             importance_mode,
#             min_node_size,
#             prediction_mode,
#             sample_with_replacement,
#             unordered_variable_names,
#             memory_saving_splitting,
#             splitrule,
#             predict_all,
#             sample_fraction,
#             alpha,
#             minprop,
#             holdout,
#             prediction_type,
#             num_random_splits,
#             order_snps,
#             max_depth,
#             regularization_factor,
#             regularization_usedepth,
#         )
#
#     def run(self, bool verbose, bool compute_oob_error):
#         dereference(self.c_fc).run(verbose, compute_oob_error)



cdef ranger(
    unsigned int treetype,
    np.ndarray[double, ndim=2, mode="fortran"] x,
    np.ndarray[double, ndim=2, mode="fortran"] y,
    vector[char*] variable_names,
    unsigned int mtry,
    unsigned int num_trees,
    bool verbose,
    unsigned int seed,
    unsigned int num_threads,
    bool write_forest,
    ranger_.ImportanceMode importance_mode_r,
    unsigned int min_node_size,
    vector[vector[double]]& split_select_weights,
    bool use_split_select_weights,
    vector[string]& always_split_variable_names,
    bool use_always_split_variable_names,
    bool prediction_mode,
    list loaded_forest,
    np.ndarray[double, ndim=2, mode="fortran"] snp_data,
    bool sample_with_replacement,
    bool probability,
    vector[string]& unordered_variable_names,
    bool use_unordered_variable_names,
    bool save_memory,
    ranger_.SplitRule splitrule_r,
    vector[double]& case_weights,
    bool use_case_weights,
    vector[double]& class_weights,
    bool predict_all,
    bool keep_inbag,
    vector[double]& sample_fraction,
    double alpha,
    double minprop,
    bool holdout,
    ranger_.PredictionType prediction_type_r,
    unsigned int num_random_splits,
    # sparse matrix sparse_x,
    bool use_sparse_data,
    bool order_snps,
    bool oob_error,
    unsigned int max_depth,
    vector[vector[size_t]]& inbag,
    bool use_inbag,
    vector[double]& regularization_factor,
    bool use_regularization_factor,
    bool regularization_usedepth,
):
    result = {}

    # cdef declarations must be at the function level
    cdef unique_ptr[ranger_.Forest] forest
    # cdef ranger_.Forest* temp
    cdef unique_ptr[ranger_.ForestClassification] temp_fc
    # cdef unique_ptr[ranger_.ForestRegression] temp_fr
    # cdef unique_ptr[ranger_.ForestProbability] temp_fp
    # cdef unique_ptr[ranger_.ForestSurvival] temp_fs
    # cdef unique_ptr[ranger_.Data] data

    cdef ranger_.ostream* verbose_out

    cdef size_t num_rows
    cdef size_t num_cols

    cdef vector[vector[vector[size_t]]] child_node_ids
    cdef vector[vector[size_t]] split_var_ids
    cdef vector[vector[double]] split_values
    cdef vector[bool] is_ordered

    cdef vector[double] class_values
    cdef const vector[double]* class_values_
    cdef vector[vector[vector[double]]] chf
    cdef vector[double] unique_timepoints
    cdef vector[vector[vector[double]]] terminal_class_counts

    cdef const vector[vector[vector[double]]] predictions

    cdef vector[vector[size_t]] snp_order

    try:
        variable_names = [str(r).encode() for r in range(x.shape[1])]

        if not use_split_select_weights:
            split_select_weights.clear()
        if not use_always_split_variable_names:
            always_split_variable_names.clear()
        if not use_unordered_variable_names:
            unordered_variable_names.clear()
        if not use_case_weights:
            case_weights.clear()
        if not use_inbag:
            inbag.clear()
        if not use_regularization_factor:
            regularization_factor.clear()

        # if verbose:
        #     # verbose_out = sys.stdout
        #     verbose_out = new ranger_.ostream()
        # else:
        #     verbose_out = new ranger_.ostream()


        # ignore sparse for now
        # if use_sparse_data:
        #     num_rows = x.shape[0]
        #     num_cols = x.shape[1]
        # else:
        #     num_rows = x.shape[0]
        #     num_cols = x.shape[1]

        # if use_sparse_data:
        #     data = unique_ptr[DataNumpy](x, y, variable_names)
        # else:
        data = DataNumpy(x, y, variable_names)

        # ignore snp data

        if treetype == ranger_.TreeType.TREE_CLASSIFICATION:
            if probability:
                # forest = make_unique[ranger_.ForestProbability]()
                pass
            else:
                forest.reset(new ranger_.ForestClassification())
        # elif treetype == ranger_.TreeType.TREE_REGRESSION:
        #     forest = make_unique[ForestRegression]()
        # elif treetype == ranger_.TreeType.TREE_SURVIVAL:
        #     forest = make_unique[ForestSurvival]()
        # elif treetype == ranger_.TreeType.TREE_PROBABILITY:
        #     forest = make_unique[ForestProbability]()

        dereference(forest).initR(
            move(data.c_data),
            mtry,
            num_trees,
            verbose_out,
            seed,
            num_threads,
            importance_mode_r,
            min_node_size,
            split_select_weights,
            always_split_variable_names,
            prediction_mode,
            sample_with_replacement,
            unordered_variable_names,
            save_memory,
            splitrule_r,
            case_weights,
            inbag,
            predict_all,
            keep_inbag,
            sample_fraction,
            alpha,
            minprop,
            holdout,
            prediction_type_r,
            num_random_splits,
            order_snps,
            max_depth,
            regularization_factor,
            regularization_usedepth,
        )

        if prediction_mode:
            child_node_ids = loaded_forest["child.nodeIDs"]
            split_var_ids = loaded_forest["split.varIDs"]
            split_values = loaded_forest["split.values"]
            is_ordered = loaded_forest["is.ordered"]

            if treetype == ranger_.TreeType.TREE_CLASSIFICATION:
                class_values = loaded_forest["class.values"]
                temp_fc = ranger_.dynamic_cast_forest_classification(forest)
                dereference(temp_fc).loadForest(num_trees, child_node_ids, split_var_ids, split_values, class_values, is_ordered)
            # elif treetype == ranger_.TreeType.TREE_REGRESSION:
            #     temp_fr = dynamic_cast[ranger_.ForestRegression](*forest)
            #     temp_fr.loadForest(num_trees, child_node_ids, split_var_ids, split_values, is_ordered)
            # elif treetype == ranger_.TreeType.TREE_SURVIVAL:
            #     chf = loaded_forest["chf"]
            #     unique_timepoints = loaded_forest["unique.death.times"]
            #     temp_fs = dynamic_cast[ranger_.ForestSurvival](*forest)
            #     temp_fs.loadForest(num_trees, child_node_ids, split_var_ids, split_values, chf, unique_timepoints, is_ordered)
            # elif treetype == ranger_.TreeType.TREE_PROBABILITY:
            #     class_values = loaded_forest["class.values"]
            #     terminal_class_counts = loaded_forest["terminal.class.counts"]
            #     temp_fp = dynamic_cast[ranger_.ForestProbability](*forest)
            #     temp_fp.loadForest(num_trees, child_node_ids, split_var_ids, split_values, class_values, terminal_class_counts, is_ordered)
        # else:
        #     if treetype == ranger_.TreeType.TREE_CLASSIFICATION and not class_weights.empty():
        #         temp_fc = dynamic_cast[ranger_.ForestClassification](*forest)
        #         dereference(temp_fc).setClassWeights(class_weights)
        #     # elif treetype == ranger_.TreeType.TREE_PROBABILITY and not class_weights.empty():
        #     #     temp_fp = dynamic_cast[ranger_.ForestProbability](*forest)
        #     #     temp_fp.setClassWeights(class_weights)
        #
        # dereference(forest).run(False, oob_error)
        #
        # if use_split_select_weights and importance_mode_r != ranger_.ImportanceMode.IMP_NONE:
        #     if verbose_out:
        #         verbose_out.write("Warning: Split select weights used. Variable importance measures are only comparable for variables with equal weights.\n", "")
        #
        # predictions = dereference(forest).getPredictions()
        # if predictions.size() == 1:
        #     if predictions[0].size() == 1:
        #         result["predictions"] = dereference(forest).getPredictions()[0][0]
        #     else:
        #         result["predictions"] = dereference(forest).getPredictions()[0]
        # else:
        #     result["predictions"] = dereference(forest).getPredictions()
        #
        # result["num_trees"] = dereference(forest).getNumTrees()
        # result["num_independent_variables"] = dereference(forest).getNumIndependentVariables()
        # # if treetype == ranger_.TreeType.TREE_SURVIVAL:
        # #     temp_fs = dynamic_cast[ranger_.ForestSurvival](*forest)
        # #     result["unique_death_times"] = temp_fs.getUniqueTimepoints()
        # if not prediction_mode:
        #     result["mtry"] = dereference(forest).getMtry()
        #     result["min_node_size"] = dereference(forest).getMinNodeSize()
        #     if importance_mode_r != ranger_.ImportanceMode.IMP_NONE:
        #         result["variable_importance"] = dereference(forest).getVariableImportance()
        #         if importance_mode_r == ranger_.ImportanceMode.IMP_PERM_CASEWISE:
        #             result["variable_importance_local"] = dereference(forest).getVariableImportanceCasewise()
        #     result["prediction_error"] = dereference(forest).getOverallPredictionError()
        #
        # if keep_inbag:
        #     result["inbag_counts"] = dereference(forest).getInbagCounts()
        #
        # if write_forest:
        #     forest_object = {
        #         "num_trees": dereference(forest).getNumTrees(),
        #         "child_node_ids": dereference(forest).getChildNodeIDs(),
        #         "splits_var_ids": dereference(forest).getSplitVarIDs(),
        #         "split_values": dereference(forest).getSplitValues(),
        #         "is_ordered": dereference(forest).getIsOrderedVariable()
        #     }
        #
        #     # if snp_data.nrow() > 1 and order_snps:
        #     #     snp_order = forest.getSnpOrder()
        #     #     forest_object["snp_order"] = vector[vector[size_t]](snp_order.begin(), snp_order.begin() + snp_data.ncol())
        #
        #     if treetype == ranger_.TreeType.TREE_CLASSIFICATION:
        #         temp_fc = dynamic_cast[ranger_.ForestClassification](*forest)
        #         class_values_ = dereference(temp_fc).getClassValues()
        #         forest_object["class_values"] = []
        #         for c in class_values_[:class_values_.size()]:
        #             forest_object["class_values"].append(c)
        #     # elif treetype == ranger_.TreeType.TREE_PROBABILITY:
        #     #     temp_fp = dynamic_cast[ranger_.ForestProbability](*forest)
        #     #     forest_object["class_values"] = temp_fp.getClassValues()
        #     #     forest_object["terminal_class_counts"] = temp_fp.getTerminalClassCounts()
        #     # elif treetype == ranger_.TreeType.TREE_SURVIVAL:
        #     #     temp_fs = dynamic_cast[ranger_.ForestSurvival](*forest)
        #     #     forest_object["chf"] = temp_fs.getChf()
        #     #     forest_object["unique_death_times"] = temp_fs.getUniqueTimepoints()
        #     result["forest"] = forest_object
        #
        # if not verbose:
        #     del verbose_out

    except KeyboardInterrupt as exc:
        return result
    except Exception as exc:
        print(exc)
        return result

    return result
