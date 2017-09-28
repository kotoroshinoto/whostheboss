import pickle
from typing import Tuple, List
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
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model


class CombinedModels(BaseEstimator, ClassifierMixin):
    def __init__(self, trained_simple_sgd_path, trained_simple_bayes_path, trained_nn_clf_path, all_codes_path, target_level=1, emptyset_label: str=None):
        self.target_level = target_level
        self.num_features = 0
        # models
        self.trained_simple_sgd = SimpleModel.load_from_pickle(trained_simple_sgd_path, is_path=True)  # type: SimpleModel
        self.trained_simple_bayes = SimpleModel.load_from_pickle(trained_simple_bayes_path, is_path=True)  # type: SimpleModel
        self.trained_nn_clf = ANNclassifier.load_from_pickle(trained_nn_clf_path)

        self.final_nn = None  # type: Sequential

    def _construct_neural_network(self):
        pass

    def assemble_features(self, X):
        #get proba predictions from each model
        #append to one giant matrix
        #return that matrix
        pass

    def preprocess_yval(self, y):
        #one-hot-encoding style
        pass

    def fit(self, X, y, **fit_params):
        self.fit_predict(X=X, y=y, fit_params=fit_params)
        return self

    def fit_predict(self, X, y, **fit_params):
        #construct neural network
        self.final_nn = self._construct_neural_network()
        #assemble features
        proba_X = self.assemble_features(X)
        #encode y values for neural network
        enc_y = self.preprocess_yval(y)
        #fit on features & y values
        self.final_nn.fit()

    def fit_titleset(self, tset: 'TitleSet'):
        set_with_empty = self.handle_emptyset_trans(tset)
        tvec = set_with_empty.get_title_vec()
        cvec = set_with_empty.get_code_vec(target_level=self.target_level)
        return self.fit(X=tvec, y=cvec)

    def predict(self, X):
        retval = self.final_nn.predict(X=self.assemble_features(X=X))
        return retval

    def predict_titleset(self, tset: 'TitleSet'):
        tvec = tset.get_title_vec()
        return self.predict(X=tvec)
