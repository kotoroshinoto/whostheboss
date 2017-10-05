import gc
import os
import shutil
from threading import Semaphore
from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from scribe_classifier.data.NOCdb.models.ensemble_util.ensemble_funcs import ObjectPickler, condense_proba_to_target_level, get_proba_sum_at_level
from scribe_classifier.data.NOCdb.readers import CodeSet
from scribe_classifier.data.NOCdb.readers import TitleSet


class CombinedModels(BaseEstimator, ClassifierMixin):
    """This object combines multiple input models in an ensemble learning effort
    to get the best prediction results possible"""
    def __init__(self,
                 all_codes: 'str',
                 emptyset_label="NA"):
        """initialize a new instance, can receive level 1, 2, and 3 dictionaries with ['sgd', 'bayes', 'neural'] inputs
        sgd and bayes are paths to SimpleModel pickles, and neural is a path to an ANNclassifier pickle.
        This object will store all of the provided models and use them as appropriate during predictions"""
        self.emptyset_label = emptyset_label
        self.num_models = 0.0
        # models
        self.models = dict()
        self.ac = CodeSet.load_from_pickle(all_codes, is_path=True)
        self.ac.add_emptyset()
        self.codes = dict()
        self.bin_encs = dict()
        self.encs = dict()  # type: Dict[int, LabelEncoder]
        self.ids = dict()
        self.lock = Semaphore()
        self.model_types = list()
        for i in range(1, 5):
            self.codes[i] = self.ac.get_codes_for_level(target_level=i)
            self.encs[i] = LabelEncoder().fit(self.codes[i])
            self.bin_encs[i] = LabelBinarizer().fit(self.codes[i])
            self.ids[i] = self.encs[i].transform(self.codes[i])
            self.models[i] = None

    def add_model(self, model, model_type, target_level):
        """For this function, the model is already loaded, it is not a path"""
        if model_type not in self.model_types:
            self.model_types.append(model_type)
        if self.models[target_level] is None:
            self.models[target_level] = dict()
        if model_type in self.models[target_level]:
            raise ValueError("You already added a model of that type for that level")
        else:
            self.models[target_level][model_type] = model
        self.num_models += 1

    def add_models_for_level(self, models, target_level):
        if models is not None:
            for model_type in models:
                self.add_model(model_type=model_type, model=models[model_type], target_level=target_level)

    def add_models_from_dict(self, models):
        """input is a Dict[int, Dict[str, obj]] indexed first by level and then by model type"""
        for i in range(1, 5):
            if i in models:
                self.add_models_for_level(models=models[i], target_level=i)

    def calc_num_models(self, target_level):
        """calculate the number of models available that are applicable for target level"""
        count = 0
        for i in range(target_level, 5):
            mdl_lvl_d = self.models[i]
            if mdl_lvl_d is None:
                continue
            for mdltype in self.model_types:
                if mdltype in mdl_lvl_d:
                    count += 1
        return count

    def _generate_proba_for_return(self, clf, X, keras_batch_size=32, is_keras=False):
        """generate probabilities for target level for given input vector on given model, and a place to store the results
        will supply keras batch size to keras model, but has to be told that it has been given a keras model"""
        if is_keras:
            proba = clf.predict_proba(X, batch_size=keras_batch_size)
        else:
            proba = clf.predict_proba(X)
        return proba

    def _generate_proba_for_serialize(self, clf, X, cachepath, keras_batch_size=32, is_keras=False):
        """generate probabilities for target level for given input vector on given model, and a place to store the results
        will supply keras batch size to keras model, but has to be told that it has been given a keras model"""
        proba = self._generate_proba_for_return(clf=clf,
                                                X=X,
                                                keras_batch_size=keras_batch_size,
                                                is_keras=is_keras)
        ObjectPickler.pickle_object(proba, cachepath)
        del proba
        proba = None
        gc.collect()

    def _generate_proba_for_level(self, proba_level: int, X, keras_batch_size=32):
        """generate probabilities for a specific model level for given input vector
         stores results in pickle form in a tmp directory"""
        mdls_dict = self.models[proba_level]
        proba_paths = dict()
        basepath = 'tmp/proba.%d.%s.P'
        if mdls_dict is None:
            return None
        else:
            for mdltype in self.model_types:
                if mdltype in mdls_dict:
                    prob_path = basepath % (proba_level, mdltype)
                    self._generate_proba_for_serialize(clf=mdls_dict[mdltype],
                                                       X=X,
                                                       cachepath=prob_path,
                                                       keras_batch_size=keras_batch_size,
                                                       is_keras=('neural' == mdltype)
                                                       )
                    proba_paths[mdltype] = prob_path
        return proba_paths

    def predict_proba(self, X, target_level=1, keras_batch_size=32) -> 'np.ndarray':
        assert self.num_models > 0.0
        """predict probabilities for input titles at targeted label levels, will pass keras batch size to neural nets
        all other predict functions funnel through this one. This function utilizes a semaphore to make sure multiple
        threads do not attempt to use it at the same time, as it serializes to disk during its operations."""
        self.lock.acquire()
        basepath = 'tmp/proba.%d.%s.P'
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
            if self.models[i] is not None:
                self._generate_proba_for_level(proba_level=i, X=X, keras_batch_size=keras_batch_size)
        for i in range(target_level, 5):
            if self.models[i] is not None:
                level_sums.append(
                    condense_proba_to_target_level(get_proba_sum_at_level(
                        proba_level=i,
                        basepath=basepath,
                        models=self.models
                    ),
                        proba_level=i,
                        target_level=target_level,
                        code_set=self.ac)
                )
        if os.path.exists('tmp'):
            if os.path.isdir('tmp'):
                shutil.rmtree('tmp', ignore_errors=True)
            else:
                os.remove('tmp')
        self.lock.release()
        return sum(level_sums) / mdl_count

    def predict_proba_per_model(self, X, keras_batch_size=32):
        """predict probabilities for input titles at targeted label levels, once for each model, returning matrices in dict
        will pass keras batch size to neural nets all other predict functions funnel through this one. This function
        utilizes a semaphore to make sure multiple threads do not attempt to use it at the same time, as it serializes
        to disk during its operations."""
        self.lock.acquire()
        proba_d = dict()
        for model_type in self.model_types:
            for i in range(1, 5):
                if i in self.models and self.models[i] is not None and model_type in self.models[i]:
                    if i not in proba_d:
                        proba_d[i] = dict()
                    proba_d[i][model_type] = self._generate_proba_for_return(
                        clf=self.models[i][model_type],
                        X=X,
                        keras_batch_size=keras_batch_size,
                        is_keras=(model_type == 'ann'))
        self.lock.release()
        return proba_d

    def batched_predict_proba_per_model_to_files(self, X, path, keras_batch_size=32, batch_size=4000, file_prefix=""):
        preds = dict()
        for i in self.models:
            preds[i] = dict()
        total_count = len(X)
        for i in range(0, total_count, batch_size):
            j = min(i + batch_size, total_count)
            print("%d to %d of %d (%d%%)" % (i + 1, j, total_count, (j * 100) / total_count))
            batch = X[i:j]
            # print('i:', i, "\tj:", j)
            proba_d = self.predict_proba_per_model(X=batch, keras_batch_size=keras_batch_size)
            for i in range(1, 5):
                if i not in proba_d or proba_d[i] is None:
                    continue
                for model_type in proba_d[i]:
                    if model_type not in preds[i]:
                        preds[i][model_type] = []
                    preds[i][model_type].append(proba_d[i][model_type])
            if j == len(X):
                break
        #recombine batched ndarrays of preds to single ndarrays and write them to disk
        for i in range(1, 5):
            if i not in preds or preds[i] is None:
                continue
            for model_type in preds[i]:
                comb_preds = np.concatenate(tuple(preds[i][model_type]))
                if len(file_prefix):
                    obj_path = os.path.join(path, file_prefix + (".proba.%d.%s.P" % (i, model_type)))
                else:
                    obj_path = os.path.join(path, "proba.%d.%s.P" % (i, model_type))
                ObjectPickler.pickle_object(obj=comb_preds, path=obj_path)

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


