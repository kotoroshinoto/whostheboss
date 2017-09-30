import pickle
import gc
import numpy as np
from typing import Tuple, List, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from scribe_classifier.data.canada.NOCdb.models.simple_model import SimpleModel
from scribe_classifier.data.canada.NOCdb.readers import TitleSet, TitleRecord
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from scribe_classifier.data.canada.NOCdb.readers import AllCodes
from .artificial_neural_net import ANNclassifier
import os
import shutil
from threading import Semaphore
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from scipy import sparse
# from keras.models import Sequential


class CombinedModels(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 all_codes: 'str',
                 lvl1_mdls: 'Dict[str, str]'=None,
                 lvl2_mdls: 'Dict[str, str]'=None,
                 lvl3_mdls: 'Dict[str, str]'=None,
                 emptyset_label="NA"):
        self.emptyset_label = emptyset_label
        self.num_models = 0.0
        # models
        self.mdls = dict()
        self.ac = AllCodes.load_from_pickle(all_codes, is_path=True)
        self.ac.add_emptyset()
        self.codes = dict()
        self.bin_encs = dict()
        self.encs = dict()  # type: Dict[int, LabelEncoder]
        self.ids = dict()
        self.lock = Semaphore()
        for i in range(1, 4):
            self.codes[i] = self.ac.get_codes_for_level(target_level=i)
            self.encs[i] = LabelEncoder().fit(self.codes[i])
            self.bin_encs[i] = LabelBinarizer().fit(self.codes[i])
            self.ids[i] = self.encs[i].transform(self.codes[i])
        if lvl1_mdls is not None:
            self.mdls[1] = dict()
            if 'sgd' in lvl1_mdls:
                self.num_models += 1.0
                self.mdls[1]['sgd'] = SimpleModel.load_from_pickle(lvl1_mdls['sgd'], is_path=True)  # type: SimpleModel
            if 'bayes' in lvl1_mdls:
                self.num_models += 1.0
                self.mdls[1]['bayes'] = SimpleModel.load_from_pickle(lvl1_mdls['bayes'], is_path=True)  # type: SimpleModel
            if 'ann' in lvl1_mdls:
                self.num_models += 1.0
                self.mdls[1]['ann'] = ANNclassifier.load_from_pickle(lvl1_mdls['ann'])  # type: ANNclassifier
        else:
            self.mdls[1] = None
        if lvl2_mdls is not None:
            self.mdls[2] = dict()
            if 'sgd' in lvl2_mdls:
                self.num_models += 1.0
                self.mdls[2]['sgd'] = SimpleModel.load_from_pickle(lvl2_mdls['sgd'], is_path=True)  # type: SimpleModel
            if 'bayes' in lvl2_mdls:
                self.num_models += 1.0
                self.mdls[2]['bayes'] = SimpleModel.load_from_pickle(lvl2_mdls['bayes'], is_path=True)  # type: SimpleModel
            if 'ann' in lvl2_mdls:
                self.num_models += 1.0
                self.mdls[2]['ann'] = ANNclassifier.load_from_pickle(lvl2_mdls['ann'])  # type: ANNclassifier
        else:
            self.mdls[2] = None
        if lvl3_mdls is not None:
            self.mdls[3] = dict()
            if 'sgd' in lvl3_mdls:
                self.num_models += 1.0
                self.mdls[3]['sgd'] = SimpleModel.load_from_pickle(lvl3_mdls['sgd'], is_path=True)  # type: SimpleModel
            if 'bayes' in lvl3_mdls:
                self.num_models += 1.0
                self.mdls[3]['bayes'] = SimpleModel.load_from_pickle(lvl3_mdls['bayes'], is_path=True)  # type: SimpleModel
            if 'ann' in lvl3_mdls:
                self.num_models += 1.0
                self.mdls[3]['ann'] = ANNclassifier.load_from_pickle(lvl3_mdls['ann'])  # type: ANNclassifier
        else:
            self.mdls[3] = None
        assert self.num_models > 0.0

    def calc_num_models(self, target_level):
        count = 0
        for i in range(target_level, 4):
            mdl_lvl_d = self.mdls[i]
            if mdl_lvl_d is None:
                continue
            for mdltype in ['sgd', 'bayes', 'ann']:
                if mdltype in mdl_lvl_d:
                    count += 1
        return count

    def generate_proba_for(self, clf, X, cachepath, keras_batch_size=32, is_keras=False):
        if is_keras:
            proba = clf.predict_proba(X, batch_size=keras_batch_size)
        else:
            proba = clf.predict_proba(X)
        ObjectPickler.pickle_object(proba, cachepath)
        del proba
        proba = None
        gc.collect()

    def generate_proba_for_level(self, proba_level: int, X, keras_batch_size=32):
        mdls_dict = self.mdls[proba_level]
        proba_paths = dict()
        basepath = 'tmp/proba.%d.%s.P'
        if mdls_dict is None:
            return None
        else:
            for mdltype in ['sgd', 'bayes', 'ann']:
                if mdltype in mdls_dict:
                    prob_path = basepath % (proba_level, mdltype)
                    self.generate_proba_for(clf=mdls_dict[mdltype],
                                            X=X,
                                            cachepath=prob_path,
                                            keras_batch_size=keras_batch_size,
                                            is_keras=('ann' == mdltype)
                                            )
                    proba_paths[mdltype] = prob_path
        return proba_paths

    def get_proba_sum_at_level(self, proba_level: int, X):
        probas = None
        mdls_dict = self.mdls[proba_level]
        basepath = 'tmp/proba.%d.%s.P'
        if mdls_dict is None:
            return None
        else:
            for mdltype in ['sgd', 'bayes', 'ann']:
                if mdltype in mdls_dict:
                    prob_path = basepath % (proba_level, mdltype)
                    reqprobas = ObjectPickler.load_object(prob_path)  # type: np.ndarray
                    if probas is None:
                        probas = reqprobas
                    else:
                        probas += reqprobas
        return probas

    def get_index_map_for_condensing(self, proba_level, target_level):
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

    def condense_proba_to_target_level(self, proba, proba_level, target_level):
        ind_map = self.get_index_map_for_condensing(proba_level, target_level=target_level)
        new_probas = np.zeros((proba.shape[0], len(self.codes[target_level])), dtype=np.float32)
        new_cols = []
        for i in range(len(ind_map)):
            new_cols.append(np.sum(np.take(proba, ind_map[i], axis=1), axis=1))
        return np.column_stack(new_cols)

    def predict_proba(self, X, target_level=1, keras_batch_size=32) -> 'np.ndarray':
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
        for i in range(target_level, 4):
            if self.mdls[i] is not None:
                self.generate_proba_for_level(proba_level=i, X=X, keras_batch_size=keras_batch_size)
        for i in range(target_level, 4):
            if self.mdls[i] is not None:
                level_sums.append(self.condense_proba_to_target_level(self.get_proba_sum_at_level(i, X), i, target_level=target_level))
        if os.path.exists('tmp'):
            if os.path.isdir('tmp'):
                shutil.rmtree('tmp', ignore_errors=True)
            else:
                os.remove('tmp')
        self.lock.release()
        return sum(level_sums) / mdl_count

    def predict(self, X, target_level=1, keras_batch_size=32):
        return self.bin_encs[target_level].inverse_transform(self.predict_proba(X=X,
                                                                                     keras_batch_size=keras_batch_size,
                                                                                     target_level=target_level))

    def predict_titleset(self, tset: 'TitleSet', target_level=1, keras_batch_size=32):
        tvec = tset.get_title_vec()
        return self.predict(X=tvec, keras_batch_size=keras_batch_size, target_level=target_level)

    def batched_predict_titleset(self, tset: 'TitleSet', target_level=1, batch_size=4000, keras_batch_size=32):
        tvec = tset.get_title_vec()
        return self.batched_predict(X=tvec, batch_size=batch_size, keras_batch_size=keras_batch_size, target_level=target_level)

    def batched_predict(self, X, target_level=1, batch_size=4000, keras_batch_size=32):
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
    @staticmethod
    def pickle_object(obj, path):
        pickle.dump(file=open(path, 'wb'), obj=obj)

    @staticmethod
    def load_object(path):
        return pickle.load(open(path, 'rb'))
