import pickle
from typing import Tuple, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from .simple_model import SimpleModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ..readers import TitleSet, TitleRecord
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from ..readers import AllCodes
from sklearn.neural_network import MLPClassifier


class CombinedModels(BaseEstimator, ClassifierMixin):
    def __init__(self, trained_simple_sgd: 'SimpleModel', trained_simple_multinom_nb: 'SimpleModel', all_codes: 'AllCodes', target_level=1, emptyset_label: str=None):
        self.target_level = target_level
        # print("combined model target_level: %d" % self.target_level)
        self.code_vec = sorted(all_codes.get_codes_for_level(target_level=self.target_level))
        self.emptyset_label=emptyset_label
        if self.emptyset_label is not None:
            self.code_vec.append(emptyset_label)
        self.label_encoder = LabelEncoder()
        # print("training encoder on: %s" % ", ".join(self.code_vec))
        self.label_encoder.fit(self.code_vec)
        self.trained_simple_sgd = trained_simple_sgd  # type: SimpleModel
        self.trained_simple_multinom_nb = trained_simple_multinom_nb  # type: SimpleModel
        self.sgdvect = self.trained_simple_sgd.clf.best_estimator_.named_steps.vect  # type: CountVectorizer
        self.nbvect = self.trained_simple_multinom_nb.clf.best_estimator_.named_steps.vect   # type: CountVectorizer
        # self.parameters = {
        #     'clf__alpha': (1e-2, 1e-5),
        #     'clf__max_iter': (1000, 3000),
        #     'clf__tol': (1e-3, 1e-4)
        # }
        self.final_nn = MLPClassifier(activation='logistic',
                                      solver='lbfgs',
                                      alpha=1e-4,
                                      max_iter=1000,
                                      tol=1e-4,
                                      verbose=True)
        # self.clf_pipe = Pipeline([
        #     ('clf', self.final_nn)
        # ])
        # self.final_clf = GridSearchCV(self.clf_pipe, self.parameters, n_jobs=-1)

    def handle_emptyset_trans(self, tset: 'TitleSet'):
        class_counts = tset.count_classes()
        if self.emptyset_label is not None:
            # if not self.use_bayes:
            prop_records = 1.0 / float(len(class_counts))
            working_title_set = tset.copy_and_append_empty_string_class(label=self.emptyset_label,
                                                                        prop_records=prop_records)
        else:
            working_title_set = tset
        return tset

    def assemble_features(self, X):
        #collect features
        sgd_bag = self.sgdvect.transform(X)
        nb_bag = self.nbvect.transform(X)
        sgd_predict = sparse.csr_matrix(self.label_encoder.transform(self.trained_simple_sgd.clf.predict(X=X)))
        nb_proba = sparse.csr_matrix(self.trained_simple_multinom_nb.clf.predict_proba(X=X))
        # print(sgd_bag.get_shape())
        # print(nb_bag.get_shape())
        # print(sgd_predict.get_shape())
        # print(nb_proba.get_shape())
        return sparse.hstack((sgd_bag, nb_bag, sgd_predict.transpose(), nb_proba))

    def fit(self, X, y, **fit_params):
        #feed features to final sgd
        # print("classes fit on: %s" % ", ".join(y))
        self.final_nn.fit(X=self.assemble_features(X=X), y=y)
        return self

    def fit_titleset(self, tset: 'TitleSet'):
        set_with_empty = self.handle_emptyset_trans(tset)
        tvec = set_with_empty.get_title_vec()
        cvec = set_with_empty.get_code_vec(target_level=self.target_level)
        return self.fit(X=tvec, y=cvec)

    def predict(self, X):
        return self.final_nn.predict(X=self.assemble_features(X=X))

    def predict_titleset(self, tset: 'TitleSet'):
        tvec = tset.get_title_vec()
        return self.predict(X=tvec)
