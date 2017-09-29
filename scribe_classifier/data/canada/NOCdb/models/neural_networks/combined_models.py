import pickle
from typing import Tuple, List, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from scribe_classifier.data.canada.NOCdb.models.simple_model import SimpleModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scribe_classifier.data.canada.NOCdb.readers import TitleSet, TitleRecord
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from scribe_classifier.data.canada.NOCdb.readers import AllCodes
from .artificial_neural_net import ANNclassifier
# from keras.layers import Dense, Dropout
# from keras.models import Sequential, load_model


class CombinedModels(BaseEstimator, ClassifierMixin):
    def __init__(self, all_codes: 'str', lvl1_mdls: 'Dict[str, str]'=None, lvl2_mdls: 'Dict[str, str]'=None, lvl3_mdls: 'Dict[str, str]'=None, target_level=1, emptyset_label="NA"):
        self.target_level = target_level
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
        for i in range(1, 4):
            self.codes[i] = self.ac.get_codes_for_level(target_level=i)
            self.encs[i] = LabelEncoder().fit(self.codes[i])
            self.bin_encs[i] = LabelBinarizer().fit(self.codes[i])
            self.ids[i] = self.encs[i].transform(self.codes[i])
        if lvl1_mdls is not None and target_level == 1:
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
        if lvl2_mdls is not None and target_level <= 2:
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
        if lvl3_mdls is not None and target_level <= 3:
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

    def get_proba_sum_at_level(self, proba_level: int, X):
        probas = []
        mdls_dict = self.mdls[proba_level]
        if mdls_dict is None:
            return None
        else:
            if 'sgd' in mdls_dict:
                probas.append(mdls_dict['sgd'].predict_proba(X))
            if 'bayes' in mdls_dict:
                probas.append(mdls_dict['bayes'].predict_proba(X))
            if 'ann' in mdls_dict:
                probas.append(mdls_dict['ann'].predict_proba(X))
        return sum(probas)

    def get_index_map_for_condensing(self, proba_level):
        proba_codes = self.ac.get_codes_for_level(target_level=proba_level)
        target_codes = self.ac.get_codes_for_level(target_level=self.target_level)
        keep_where = []
        for i in range(len(target_codes)):
            keep_where_row = []
            targ_label = target_codes[i]
            for j in range(len(proba_codes)):
                prob_label = proba_codes[j]
                if prob_label != self.emptyset_label:
                    prob_label = prob_label[0: self.target_level]
                if prob_label == targ_label:
                    keep_where_row.append(j)
            keep_where.append(keep_where_row)
        return keep_where

    def condense_proba_to_target_level(self, proba, proba_level):
        ind_map = self.get_index_map_for_condensing(proba_level)
        new_probas = np.zeros((proba.shape[0], len(self.codes[self.target_level])), dtype=np.float32)
        new_cols = []
        for i in range(len(ind_map)):
            new_cols.append(np.sum(np.take(proba, ind_map[i], axis=1), axis=1))
        return np.column_stack(new_cols)

    def predict_proba(self, X) -> 'np.ndarray':
        level_sums = list()
        for i in range(1, 4):
            if self.mdls[i] is not None:
                level_sums.append(self.condense_proba_to_target_level(self.get_proba_sum_at_level(i, X), i))
        return sum(level_sums) / self.num_models

    def predict(self, X):
        return self.bin_encs[self.target_level].inverse_transform(self.predict_proba(X))

    def predict_titleset(self, tset: 'TitleSet'):
        tvec = tset.get_title_vec()
        return self.predict(X=tvec)
