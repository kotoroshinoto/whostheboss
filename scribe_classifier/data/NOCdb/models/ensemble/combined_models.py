import gc
import os
import pickle
import shutil
import numpy as np
from threading import Semaphore
from typing import Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from scribe_classifier.data.NOCdb.models.neural_networks.artificial_neural_net import ANNclassifier
from scribe_classifier.data.NOCdb.models.simple.simple_model import SimpleModel
from scribe_classifier.data.NOCdb.readers import CodeSet
from scribe_classifier.data.NOCdb.readers import TitleSet


class CombinedModels(BaseEstimator, ClassifierMixin):
    """This object combines multiple input models in an ensemble learning effort
    to get the best prediction results possible"""
    def __init__(self,
                 all_codes: 'str',
                 lvl1_mdls: 'Dict[str, str]'=None,
                 lvl2_mdls: 'Dict[str, str]'=None,
                 lvl3_mdls: 'Dict[str, str]'=None,
                 lvl4_mdls: 'Dict[str, str]'=None,
                 emptyset_label="NA"):
        """initialize a new instance, can receive level 1, 2, and 3 dictionaries with ['sgd', 'bayes', 'neural'] inputs
        sgd and bayes are paths to SimpleModel pickles, and neural is a path to an ANNclassifier pickle.
        This object will store all of the provided models and use them as appropriate during predictions"""
        self.emptyset_label = emptyset_label
        self.num_models = 0.0
        # models
        self.mdls = dict()
        self.ac = CodeSet.load_from_pickle(all_codes, is_path=True)
        self.ac.add_emptyset()
        self.codes = dict()
        self.bin_encs = dict()
        self.encs = dict()  # type: Dict[int, LabelEncoder]
        self.ids = dict()
        self.lock = Semaphore()
        for i in range(1, 5):
            self.codes[i] = self.ac.get_codes_for_level(target_level=i)
            self.encs[i] = LabelEncoder().fit(self.codes[i])
            self.bin_encs[i] = LabelBinarizer().fit(self.codes[i])
            self.ids[i] = self.encs[i].transform(self.codes[i])
        self._add_models_for_level(lvl_mdls=lvl1_mdls, target_level=1)
        self._add_models_for_level(lvl_mdls=lvl2_mdls, target_level=2)
        self._add_models_for_level(lvl_mdls=lvl3_mdls, target_level=3)
        self._add_models_for_level(lvl_mdls=lvl4_mdls, target_level=4)
        assert self.num_models > 0.0

    def _add_models_for_level(self, lvl_mdls, target_level):
        """Adds models for a given level"""
        if lvl_mdls is not None:
            self.mdls[target_level] = dict()
            for mdltype in ['sgd', 'bayes', 'neural']:
                if mdltype in lvl_mdls:
                    # print("adding 1 to lvl_mdls")
                    self.num_models += 1.0
                    if mdltype == 'neural':
                        self.mdls[target_level][mdltype] = ANNclassifier.load_from_pickle(
                            filepath=lvl_mdls[mdltype]
                        )  # type: ANNclassifier
                    else:
                        self.mdls[target_level][mdltype] = SimpleModel.load_from_pickle(
                            file=lvl_mdls[mdltype],
                            is_path=True
                        )  # type: SimpleModel
        else:
            self.mdls[target_level] = None

    def calc_num_models(self, target_level):
        """calculate the number of models available that are applicable for target level"""
        count = 0
        for i in range(target_level, 5):
            mdl_lvl_d = self.mdls[i]
            if mdl_lvl_d is None:
                continue
            for mdltype in ['sgd', 'bayes', 'neural']:
                if mdltype in mdl_lvl_d:
                    count += 1
        return count

    def _generate_proba_for(self, clf, X, cachepath, keras_batch_size=32, is_keras=False):
        """generate probabilities for target level for given input vector on given model, and a place to store the results
        will supply keras batch size to keras model, but has to be told that it has been given a keras model"""
        if is_keras:
            proba = clf.predict_proba(X, batch_size=keras_batch_size)
        else:
            proba = clf.predict_proba(X)
        ObjectPickler.pickle_object(proba, cachepath)
        del proba
        proba = None
        gc.collect()

    def _generate_proba_for_level(self, proba_level: int, X, keras_batch_size=32):
        """generate probabilities for a specific model level for given input vector
         stores results in pickle form in a tmp directory"""
        mdls_dict = self.mdls[proba_level]
        proba_paths = dict()
        basepath = 'tmp/proba.%d.%s.P'
        if mdls_dict is None:
            return None
        else:
            for mdltype in ['sgd', 'bayes', 'neural']:
                if mdltype in mdls_dict:
                    prob_path = basepath % (proba_level, mdltype)
                    self._generate_proba_for(clf=mdls_dict[mdltype],
                                             X=X,
                                             cachepath=prob_path,
                                             keras_batch_size=keras_batch_size,
                                             is_keras=('neural' == mdltype)
                                             )
                    proba_paths[mdltype] = prob_path
        return proba_paths

    def _get_proba_sum_at_level(self, proba_level: int, X):
        """read previously produced pickle files containing probability matrix,
         and sum all the probability matrices into a single matrix, returning result"""
        probas = None
        mdls_dict = self.mdls[proba_level]
        basepath = 'tmp/proba.%d.%s.P'
        if mdls_dict is None:
            return None
        else:
            for mdltype in ['sgd', 'bayes', 'neural']:
                if mdltype in mdls_dict:
                    prob_path = basepath % (proba_level, mdltype)
                    reqprobas = ObjectPickler.load_object(prob_path)  # type: np.ndarray
                    if probas is None:
                        probas = reqprobas
                    else:
                        probas += reqprobas
        return probas

    def _get_index_map_for_condensing(self, proba_level, target_level):
        """gets a map of indexes for combining probability produced for a given level
         into an equivalent matrix for target level"""
        proba_codes = self.ac.get_codes_for_level(target_level=proba_level)
        target_codes = self.ac.get_codes_for_level(target_level=target_level)
        keep_where = []
        for i in range(len(target_codes)):
            keep_where_row = []
            targ_label = target_codes[i]
            for j in range(len(proba_codes)):
                prob_label = proba_codes[j]
                if prob_label == self.emptyset_label:
                    prob_label = self.emptyset_label
                else:
                    prob_label = prob_label[0: target_level]
                if prob_label == targ_label:
                    keep_where_row.append(j)
            keep_where.append(keep_where_row)
        return keep_where

    def _condense_proba_to_target_level(self, proba, proba_level, target_level):
        """change probabilities for level N labels to equivalent probabilities for level M labels (M<N)"""
        ind_map = self._get_index_map_for_condensing(proba_level, target_level=target_level)
        new_probas = np.zeros((proba.shape[0], len(self.codes[target_level])), dtype=np.float32)
        new_cols = []
        for i in range(len(ind_map)):
            new_cols.append(np.sum(np.take(proba, ind_map[i], axis=1), axis=1))
        return np.column_stack(new_cols)

    def predict_proba(self, X, target_level=1, keras_batch_size=32) -> 'np.ndarray':
        """predict probabilities for input titles at targeted label levels, will pass keras batch size to neural nets
        all other predict functions funnel through this one. This function utilizes a semaphore to make sure multiple
        threads do not attempt to use it at the same time, as it serializes to disk during its operations."""
        self.lock.acquire()
        if os.path.exists('tmp'):
            if os.path.isdir('tmp'):
                shutil.rmtree('tmp', ignore_errors=True)
            else:
                os.remove('tmp')
        os.mkdir('tmp')
        mdl_count = self.calc_num_models(target_level=target_level)
        if mdl_count == 0:
            raise RuntimeError("Cannot predict proba for level %d without relevant models" % target_level)
        gc.enable()
        level_sums = list()
        #pre-generate probas
        for i in range(target_level, 5):
            if self.mdls[i] is not None:
                self._generate_proba_for_level(proba_level=i, X=X, keras_batch_size=keras_batch_size)
        for i in range(target_level, 5):
            if self.mdls[i] is not None:
                level_sums.append(self._condense_proba_to_target_level(self._get_proba_sum_at_level(i, X), i, target_level=target_level))
        if os.path.exists('tmp'):
            if os.path.isdir('tmp'):
                shutil.rmtree('tmp', ignore_errors=True)
            else:
                os.remove('tmp')
        self.lock.release()
        return sum(level_sums) / mdl_count

    def predict(self, X, target_level=1, keras_batch_size=32):
        """Predict classes at requested target level for input titles. Will pass keras batch size to neural nets"""
        return self.bin_encs[target_level].inverse_transform(self.predict_proba(X=X,
                                                                                     keras_batch_size=keras_batch_size,
                                                                                     target_level=target_level))

    def predict_titleset(self, tset: 'TitleSet', target_level=1, keras_batch_size=32):
        """Convenience function for titleset objects. Will predict class codes for desired class level
        Will pass keras batch size to neural nets"""
        tvec = tset.get_title_vec()
        return self.predict(X=tvec, keras_batch_size=keras_batch_size, target_level=target_level)

    def batched_predict_titleset(self, tset: 'TitleSet', target_level=1, batch_size=4000, keras_batch_size=32):
        """Convenience function for titleset objects. Will predict class codes for desired class level, dividing into
        batches of desired size to avoid memory errors from expanding bag of words into a dense matrix. \n
        Will pass keras batch size to neural nets"""
        tvec = tset.get_title_vec()
        return self.batched_predict(X=tvec, batch_size=batch_size, keras_batch_size=keras_batch_size, target_level=target_level)

    def batched_predict(self, X, target_level=1, batch_size=4000, keras_batch_size=32):
        """Predict classes at requested target level for input titles. , dividing into batches of desired size to avoid
         memory errors from expanding bag of words into a dense matrix. Will pass keras batch size to neural nets"""
        preds = []
        total_count = len(X)
        for i in range(0, total_count, batch_size):
            j = min(i + batch_size, total_count)
            print("%d to %d of %d (%d%%)" % (i+1, j, total_count, (j*100) / total_count))
            batch = X[i:j]
            # print('i:', i, "\tj:", j)
            preds.append(self.predict(batch, target_level=target_level, keras_batch_size=keras_batch_size))
            if j == len(X):
                break
        # for i in range(len(preds)):
        #     print(preds[i].shape)
        preds = np.concatenate(tuple(preds))
        print(preds.shape)
        return preds


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
