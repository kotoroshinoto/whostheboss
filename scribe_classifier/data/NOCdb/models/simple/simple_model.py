import pickle
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scribe_classifier.data.NOCdb.readers import TitleSet, TitleRecord


class SimpleModel(BaseEstimator, ClassifierMixin):
    """This class can learn using one of 3 methods, multinomial bayes, support vector classifiers,
    or stochastic gradient descent support vectors"""
    def __init__(self, target_level=1, emptyset_label: str=None, cv=None, ngram_start=1, ngram_stop=5, model_type='sgdsv'):
        self.target_level = target_level
        self.model_type = model_type
        self.parameters = dict()
        self.vect = CountVectorizer(stop_words='english', ngram_range=(ngram_start, ngram_stop))
        # self.parameters['vect__ngram_range'] = [(1, x) for x in range(ngram_start, ngram_stop + ngram_step, ngram_step)]

        if self.model_type == 'bayes':
            self.parameters['alpha'] = (1e-3, 1e-4, 1e-5, 1e-6)
            # self.prop_records = 1.0/8.0
            self.ml_clf = MultinomialNB(alpha=1e-4)
        elif self.model_type == 'sgdsv':
            self.parameters['alpha'] = (1e-3, 1e-4, 1e-5, 1e-6)
            self.parameters['tol'] = (1e-3, 1e-4, 1e-5, 1e-6)
            self.ml_clf = SGDClassifier(alpha=1e-4, max_iter=3000, tol=1e-4, n_jobs=-1, loss='log')
        elif self.model_type == 'svc':
            self.ml_clf = SVC(kernel='linear', probability=True)
        else:
            raise ValueError("Unrecognized model type")

        self.clf = GridSearchCV(self.ml_clf, self.parameters, n_jobs=-1, cv=cv, scoring='accuracy')
        if emptyset_label is not None:
            if emptyset_label == "":
                self.emptyset_label = "NA"
            else:
                self.emptyset_label = emptyset_label

    def save_as_pickle(self, file, is_path=False):
        """saves model as a pickle"""
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'SimpleModel':
        """load model data from pickle"""
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        smdl = pickle.load(handle)
        handle.close()
        return smdl

    def fit_titleset(self, title_set: 'TitleSet'):
        """fits model against title set"""
        class_counts = title_set.count_classes()
        if self.emptyset_label is not None:
            if self.model_type == 'bayes':
                prop_records = 0.25
            else:
                prop_records = 1.0 / float(len(class_counts))

            working_title_set = title_set.copy_and_append_empty_string_class(label=self.emptyset_label,
                                                                             prop_records=prop_records)
        else:
            working_title_set = title_set
        X, Y = working_title_set.split_into_title_and_code_vecs(target_level=self.target_level)
        return self.fit(X, Y)

    def predict_titleset(self, title_set: 'TitleSet') -> 'List[str]':
        """make predictions for given title set"""
        return self.predict(title_set.get_title_vec())

    def predict_titlevec(self, title_vec: 'List[TitleRecord]') -> 'List[str]':
        """make predictions for given title record vector"""
        return self.predict(title_vec)

    def predict_titlerecord(self, title_record: 'TitleRecord') -> str:
        """make predictions for given title record"""
        # print(type(title_record))
        return self.predict([title_record.title])[0]

    def fit(self, X, y, **fit_params):
        """fit against X input titles, given y classes"""
        bag = self.vect.fit_transform(X, y)
        self.clf.fit(bag, y)
        return self

    def predict(self, X):
        """predict classes for input X"""
        bag = self.vect.transform(X)
        return self.clf.predict(bag)

    def predict_proba(self, X):
        """predict class probabilities for input X"""
        bag = self.vect.transform(X)
        return self.clf.predict_proba(bag)

    # TODO, will probably need to implement partial fitting for batched training on very large datasets later on
