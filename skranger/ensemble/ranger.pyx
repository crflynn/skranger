"""Cython implementation for ranger and Data child classes."""
import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from skranger.ensemble cimport ranger_


cdef class DataNumpy:
    """Cython wrapper for DataNumpy C++ class in ``DataNumpy.h``.

    This wraps the Data class in C++, which encapsulates training data passed to the
    random forest classes. It allows us to pass numpy arrays as a ranger-compatible
    Data object.
    """
    cdef unique_ptr[ranger_.DataNumpy] c_data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self,
        np.ndarray[double, ndim=2, mode="fortran"] x not None,
        np.ndarray[double, ndim=2, mode="fortran"] y not None,
        vector[string] variable_names,
    ):
        cdef size_t num_rows = np.PyArray_DIMS(x)[0]  # in lieu of x.shape
        cdef size_t num_cols = np.PyArray_DIMS(x)[1]
        cdef size_t num_cols_y = np.PyArray_DIMS(y)[1]
        self.c_data.reset(
            new ranger_.DataNumpy(
                &x[0, 0],
                &y[0, 0],
                variable_names,
                num_rows,
                num_cols,
                num_cols_y,
            )
        )

    def get_x(self, size_t row, size_t col):
        return deref(self.c_data).get_x(row, col)

    def get_y(self, size_t row, size_t col):
        return deref(self.c_data).get_y(row, col)

    def reserve_memory(self, size_t y_cols):
        return deref(self.c_data).reserveMemory(y_cols)

    def set_x(self, size_t col, size_t row, double value, bool& error):
        return deref(self.c_data).set_x(col, row, value, error)

    def set_y(self, size_t col, size_t row, double value, bool& error):
        return deref(self.c_data).set_y(col, row, value, error)


cpdef dict ranger(
    ranger_.TreeType treetype,
    np.ndarray[double, ndim=2, mode="fortran"] x,
    np.ndarray[double, ndim=2, mode="fortran"] y,
    vector[string]& variable_names,
    unsigned int mtry,
    unsigned int num_trees,
    bool verbose,
    unsigned int seed,
    unsigned int num_threads,
    bool write_forest,
    ranger_.ImportanceMode importance_mode,
    unsigned int min_node_size,
    vector[vector[double]]& split_select_weights,
    bool use_split_select_weights,
    vector[string]& always_split_variable_names,
    bool use_always_split_variable_names,
    bool prediction_mode,
    dict loaded_forest,
    np.ndarray[double, ndim=2, mode="fortran"] snp_data,
    bool sample_with_replacement,
    bool probability,
    vector[string]& unordered_variable_names,
    bool use_unordered_variable_names,
    bool save_memory,
    ranger_.SplitRule splitrule,
    vector[double]& case_weights,
    bool use_case_weights,
    vector[double]& class_weights,
    bool predict_all,
    bool keep_inbag,
    vector[double]& sample_fraction,
    double alpha,
    double minprop,
    bool holdout,
    ranger_.PredictionType prediction_type,
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
    """Cython function interface to ranger.
    
    Provides an entrypoint into the ranger C++ code, and returns a result object
    with ranger-specific random forest implementation objects for serializing and
    deserializing random forests. The result object is a python dictionary containing
    forest structures, metadata, and predictions (if in prediction mode). The structure
    of results is (depending on forest type):
    
    {
        "predictions": [[]],
        "num_trees": int,
        "num_independent_variables": int,
        "mtry": int,
        "min_node_size": int,
        "prediction_error": float,
        "inbag_counts": -,
        "unique_death_times": -,
        "variable_importance": -,
        "variable_importance_local": -,
        "forest": {
            "num_trees": int,
            "child_node_ids": [[[]]],
            "split_var_ids": [[]],
            "split_values": [[]],
            "is_ordered": [],
            "class_values": [],
            "terminal_class_counts": [[[]]],
            "cumulative_hazard_function": -,
            "unique_death_times": -,
        },
    }
    """
    # print(locals())
    result = {}

    cdef unique_ptr[ranger_.Forest] forest

    cdef ranger_.ostream* verbose_out

    cdef vector[vector[vector[size_t]]] child_node_ids
    cdef vector[vector[size_t]] split_var_ids
    cdef vector[vector[double]] split_values
    cdef vector[bool] is_ordered
    cdef vector[double] class_values
    cdef vector[double] class_values_
    cdef vector[vector[vector[double]]] cumulative_hazard_function
    cdef vector[double] unique_timepoints
    cdef vector[vector[vector[double]]] terminal_class_counts
    cdef vector[vector[vector[double]]] predictions

    try:
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

        if verbose:
            verbose_out = <ranger_.ostream*> &ranger_.cout
        else:
            verbose_out = <ranger_.ostream*> new ranger_.stringstream()

        data = DataNumpy(x, y, variable_names)

        if treetype == ranger_.TreeType.TREE_CLASSIFICATION:
            if probability:
                forest.reset(new ranger_.ForestProbability())
            else:
                forest.reset(new ranger_.ForestClassification())
        elif treetype == ranger_.TreeType.TREE_REGRESSION:
            forest.reset(new ranger_.ForestRegression())
        elif treetype == ranger_.TreeType.TREE_SURVIVAL:
            forest.reset(new ranger_.ForestSurvival())
        elif treetype == ranger_.TreeType.TREE_PROBABILITY:
            forest.reset(new ranger_.ForestProbability())

        deref(forest).initR(
            move(data.c_data),
            mtry,
            num_trees,
            verbose_out,
            seed,
            num_threads,
            importance_mode,
            min_node_size,
            split_select_weights,
            always_split_variable_names,
            prediction_mode,
            sample_with_replacement,
            unordered_variable_names,
            save_memory,
            splitrule,
            case_weights,
            inbag,
            predict_all,
            keep_inbag,
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

        if prediction_mode:
            child_node_ids = loaded_forest["child_node_ids"]
            split_var_ids = loaded_forest["split_var_ids"]
            split_values = loaded_forest["split_values"]
            is_ordered = loaded_forest["is_ordered"]

            if treetype == ranger_.TreeType.TREE_CLASSIFICATION:
                class_values = loaded_forest["class_values"]
                (<ranger_.ForestClassification*> forest.get()).loadForest(num_trees, child_node_ids, split_var_ids, split_values, class_values, is_ordered)
            elif treetype == ranger_.TreeType.TREE_REGRESSION:
                (<ranger_.ForestRegression*> forest.get()).loadForest(num_trees, child_node_ids, split_var_ids, split_values, is_ordered)
            elif treetype == ranger_.TreeType.TREE_SURVIVAL:
                cumulative_hazard_function = loaded_forest["cumulative_hazard_function"]
                unique_timepoints = loaded_forest["unique_death_times"]
                (<ranger_.ForestSurvival*> forest.get()).loadForest(num_trees, child_node_ids, split_var_ids, split_values, cumulative_hazard_function, unique_timepoints, is_ordered)
            elif treetype == ranger_.TreeType.TREE_PROBABILITY:
                class_values = loaded_forest["class_values"]
                terminal_class_counts = loaded_forest["terminal_class_counts"]
                (<ranger_.ForestProbability*> forest.get()).loadForest(num_trees, child_node_ids, split_var_ids, split_values, class_values, terminal_class_counts, is_ordered)
        else:
            if treetype == ranger_.TreeType.TREE_CLASSIFICATION and not class_weights.empty():
                (<ranger_.ForestClassification*> forest.get()).setClassWeights(class_weights)
            elif treetype == ranger_.TreeType.TREE_PROBABILITY and not class_weights.empty():
                (<ranger_.ForestProbability*> forest.get()).setClassWeights(class_weights)

        deref(forest).run(verbose, oob_error)

        if use_split_select_weights and importance_mode != ranger_.ImportanceMode.IMP_NONE:
            if verbose_out:
                verbose_out.write("Warning: Split select weights used. Variable importance measures are only comparable for variables with equal weights.\n", 1)

        predictions = deref(forest).getPredictions()
        if predictions.size() == 1:
            if predictions[0].size() == 1:
                result["predictions"] = predictions[0][0]
            else:
                result["predictions"] = predictions[0]
        else:
            result["predictions"] = predictions

        result["num_trees"] = deref(forest).getNumTrees()
        result["num_independent_variables"] = deref(forest).getNumIndependentVariables()
        if treetype == ranger_.TreeType.TREE_SURVIVAL:
            result["unique_death_times"] = (<ranger_.ForestSurvival*> forest.get()).getUniqueTimepoints()
        if not prediction_mode:
            result["mtry"] = deref(forest).getMtry()
            result["min_node_size"] = deref(forest).getMinNodeSize()
            if importance_mode != ranger_.ImportanceMode.IMP_NONE:
                result["variable_importance"] = deref(forest).getVariableImportance()
                if importance_mode == ranger_.ImportanceMode.IMP_PERM_CASEWISE:
                    result["variable_importance_local"] = deref(forest).getVariableImportanceCasewise()
            result["prediction_error"] = deref(forest).getOverallPredictionError()

        if keep_inbag:
            result["inbag_counts"] = deref(forest).getInbagCounts()

        if write_forest:
            forest_object = {
                "num_trees": deref(forest).getNumTrees(),
                "child_node_ids": deref(forest).getChildNodeIDs(),
                "split_var_ids": deref(forest).getSplitVarIDs(),
                "split_values": deref(forest).getSplitValues(),
                "is_ordered": deref(forest).getIsOrderedVariable()
            }

            if treetype == ranger_.TreeType.TREE_CLASSIFICATION:
                class_values_ = (<ranger_.ForestClassification*> forest.get()).getClassValues()
                forest_object["class_values"] = []
                for c in class_values_[:class_values_.size()]:
                    forest_object["class_values"].append(c)
            elif treetype == ranger_.TreeType.TREE_PROBABILITY:
                forest_object["class_values"] = (<ranger_.ForestProbability*> forest.get()).getClassValues()
                forest_object["terminal_class_counts"] = (<ranger_.ForestProbability*> forest.get()).getTerminalClassCounts()
            elif treetype == ranger_.TreeType.TREE_SURVIVAL:
                forest_object["cumulative_hazard_function"] = (<ranger_.ForestSurvival*> forest.get()).getChf()
                forest_object["unique_death_times"] = (<ranger_.ForestSurvival*> forest.get()).getUniqueTimepoints()
            result["forest"] = forest_object

        if not verbose:
            del verbose_out

    except Exception as exc:
        raise exc

    return result
