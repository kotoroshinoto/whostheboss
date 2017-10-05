import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scribe_classifier.data.NOCdb.readers import CodeSet


class ObjectPickler:
    """helper class for pickling and unpickling objects"""
    @staticmethod
    def pickle_object(obj, path):
        """save serialized python objects to file"""
        pickle.dump(file=open(path, 'wb'), obj=obj)

    @staticmethod
    def load_object(path):
        """load serialized python objects from file"""
        return pickle.load(open(path, 'rb'))


def condense_proba_to_target_level(proba, proba_level: int, target_level: int, code_set: 'CodeSet', emptyset_label="NA"):
    """change probabilities for level N labels to equivalent probabilities for level M labels (M<N)"""
    ind_map = get_index_map_for_condensing(code_set=code_set,
                                           proba_level=proba_level,
                                           target_level=target_level,
                                           emptyset_label=emptyset_label)
    new_cols = []
    for i in range(len(ind_map)):
        new_cols.append(np.sum(np.take(proba, ind_map[i], axis=1), axis=1))
    return np.column_stack(new_cols)


def get_index_map_for_condensing(code_set: 'CodeSet', proba_level: int, target_level: int, emptyset_label="NA"):
    """gets a map of indexes for combining probability produced for a given level
     into an equivalent matrix for target level"""
    proba_codes = code_set.get_codes_for_level(target_level=proba_level)
    target_codes = code_set.get_codes_for_level(target_level=target_level)
    keep_where = []
    for i in range(len(target_codes)):
        keep_where_row = []
        targ_label = target_codes[i]
        for j in range(len(proba_codes)):
            prob_label = proba_codes[j]
            if prob_label == emptyset_label:
                prob_label = emptyset_label
            else:
                prob_label = prob_label[0: target_level]
            if prob_label == targ_label:
                keep_where_row.append(j)
        keep_where.append(keep_where_row)
    return keep_where


def get_proba_sum_at_level(models, proba_level: int, basepath):
    """read previously produced pickle files containing probability matrix,
     and sum all the probability matrices into a single matrix, returning result"""
    probas = None
    mdls_dict = models[proba_level]
    if mdls_dict is None:
        return None
    else:
        for mdltype in mdls_dict:
            prob_path = basepath % (proba_level, mdltype)
            reqprobas = ObjectPickler.load_object(prob_path)  # type: np.ndarray
            if probas is None:
                probas = reqprobas
            else:
                probas += reqprobas
    return probas


def condense_probas_at_levels(probas_d, target_level, code_set: 'CodeSet', emptyset_label="NA"):
    condensed = []
    for i in range(target_level, 5):
        if i not in probas_d:
            continue
        condensed.append(condense_proba_to_target_level(
            proba=probas_d[i],
            proba_level=i,
            target_level=target_level,
            code_set=code_set,
            emptyset_label=emptyset_label
        ))
    return condensed


def get_proba_sums_at_levels_with_prefix(basepath, prefix, target_level, include_dict):
    probas_d = dict()
    for i in range(target_level, 5):
        if i not in include_dict:
            continue
        if len(prefix):
            obj_basepath = os.path.join(basepath, prefix + ".proba.%d.%s.P")
        else:
            obj_basepath = os.path.join(basepath, "proba.%d.%s.P")
        probas_d[i] = get_proba_sum_at_level(models=include_dict, proba_level=i, basepath=obj_basepath)
    return probas_d


def predict_from_files(basepath, prefix, target_level, include_dict, code_set: 'CodeSet', emptyset_label="NA"):
    """include dict should mirror structure of models dict[int,dict[str, obj]]
    in the sense that it should have matching keys in a dict[int, list[str]]"""
    lblbin = LabelBinarizer()
    lblbin.fit(y=code_set.get_codes_for_level(target_level=target_level))
    probas = get_proba_sums_at_levels_with_prefix(basepath=basepath,
                                                  prefix=prefix,
                                                  target_level=target_level,
                                                  include_dict=include_dict)
    probas = condense_probas_at_levels(
        probas_d=probas,
        target_level=target_level,
        code_set=code_set,
        emptyset_label=emptyset_label
    )
    return lblbin.inverse_transform(sum(probas) / len(probas))