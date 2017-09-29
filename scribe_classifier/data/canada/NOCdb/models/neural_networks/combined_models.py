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
from sklearn.preprocessing import LabelEncoder
from scribe_classifier.data.canada.NOCdb.readers import AllCodes
from .artificial_neural_net import ANNclassifier
# from keras.layers import Dense, Dropout
# from keras.models import Sequential, load_model


class CombinedModels(BaseEstimator, ClassifierMixin):
    def __init__(self, all_codes: 'str', lvl1_mdls: 'Dict[str, str]'=None, lvl2_mdls: 'Dict[str, str]'=None, lvl3_mdls: 'Dict[str, str]'=None, target_level=1, emptyset_label="NA"):
        self.target_level = target_level
        self.emptyset_label = emptyset_label
        self.num_models = 0
        # models
        self.mdls = dict()
        self.ac = AllCodes.load_from_pickle(all_codes, is_path=True)
        self.ac.add_emptyset()
        self.encs = dict()  # type: Dict[int, LabelEncoder]
        self.ids = dict()
        for i in range(1, 4):
            self.encs[i] = LabelEncoder().fit(self.ac.get_codes_for_level(target_level=i))
            self.ids[i] = self.encs[i].transform(self.ac.get_codes_for_level(target_level=i))
        if lvl1_mdls is not None and target_level == 1:
            self.mdls[1] = dict()
            if 'sgd' in lvl1_mdls:
                self.num_models += 1
                self.mdls[1]['sgd'] = SimpleModel.load_from_pickle(lvl1_mdls['sgd'], is_path=True)  # type: SimpleModel
            if 'bayes' in lvl1_mdls:
                self.num_models += 1
                self.mdls[1]['bayes'] = SimpleModel.load_from_pickle(lvl1_mdls['bayes'], is_path=True)  # type: SimpleModel
            if 'ann' in lvl1_mdls:
                self.num_models += 1
                self.mdls[1]['ann'] = ANNclassifier.load_from_pickle(lvl1_mdls['ann'])  # type: ANNclassifier
        else:
            self.mdls[1] = None
        if lvl2_mdls is not None and target_level <= 2:
            self.mdls[2] = dict()
            if 'sgd' in lvl2_mdls:
                self.num_models += 1
                self.mdls[2]['sgd'] = SimpleModel.load_from_pickle(lvl2_mdls['sgd'], is_path=True)  # type: SimpleModel
            if 'bayes' in lvl2_mdls:
                self.num_models += 1
                self.mdls[2]['bayes'] = SimpleModel.load_from_pickle(lvl2_mdls['bayes'], is_path=True)  # type: SimpleModel
            if 'ann' in lvl2_mdls:
                self.num_models += 1
                self.mdls[2]['ann'] = ANNclassifier.load_from_pickle(lvl2_mdls['ann'])  # type: ANNclassifier
        else:
            self.mdls[2] = None
        if lvl3_mdls is not None and target_level <= 3:
            self.mdls[3] = dict()
            if 'sgd' in lvl3_mdls:
                self.num_models += 1
                self.mdls[3]['sgd'] = SimpleModel.load_from_pickle(lvl3_mdls['sgd'], is_path=True)  # type: SimpleModel
            if 'bayes' in lvl3_mdls:
                self.num_models += 1
                self.mdls[3]['bayes'] = SimpleModel.load_from_pickle(lvl3_mdls['bayes'], is_path=True)  # type: SimpleModel
            if 'ann' in lvl3_mdls:
                self.num_models += 1
                self.mdls[3]['ann'] = ANNclassifier.load_from_pickle(lvl3_mdls['ann'])  # type: ANNclassifier
        else:
            self.mdls[3] = None
        assert self.num_models > 0

    def sum_relevant_probas(self, proba, proba_level):
        # print("target_level: ", self.target_level)
        # print("proba_level: ", proba_level)
        proba_ids = self.ids[proba_level]
        proba_enc = self.encs[proba_level]
        target_enc = self.encs[self.target_level]
        target_ids = self.ids[self.target_level]
        new_probas = np.zeros((proba.shape[0], len(target_ids)), dtype=np.float32)
        for i in range(proba.shape[1]):
            # figure out columns included for given label and get indices for slice
            np.split
        for i in range(proba.shape[0]):
            #slice columns and get their rowsums
            for j in range(proba.shape[1]):
                label_at_pos = proba_enc.inverse_transform([j])[0]
                # print("label at level: ", label_at_pos)
                if label_at_pos == self.emptyset_label:
                    target_level_label = self.emptyset_label
                else:
                    target_level_label = label_at_pos[0:self.target_level]
                # print("target label: ", target_level_label)
                new_probas[i][target_enc.transform([target_level_label])[0]] += proba[i][j]
        return new_probas

    def get_proba_for_level(self, req_level: int, X):
        probas = []
        mdls_dict = self.mdls[req_level]
        if mdls_dict is None:
            return None
        else:
            if 'sgd' in mdls_dict:
                self.sum_relevant_probas(probas.append(mdls_dict['sgd'].predict_proba(X)), req_level)
            if 'bayes' in mdls_dict:
                self.sum_relevant_probas(probas.append(mdls_dict['bayes'].predict_proba(X)), req_level)
            if 'ann' in mdls_dict:
                self.sum_relevant_probas(probas.append(mdls_dict['ann'].predict_proba(X)), req_level)

    def predict_proba(self, X) -> 'np.ndarray':
        #get proba predictions from each model
        sgd_proba = self.trained_simple_sgd.predict_proba(X)
        bayes_proba = self.trained_simple_bayes.predict_proba(X)
        nn_proba = self.trained_nn_clf.predict_proba(X)
        #get average of proba
        return (sgd_proba + bayes_proba + nn_proba) / self.num_models

    def predict(self, X):
        return self.trained_nn_clf.lbl_bin.inverse_transform(self.predict_proba(X))

    def predict_titleset(self, tset: 'TitleSet'):
        tvec = tset.get_title_vec()
        return self.predict(X=tvec)
