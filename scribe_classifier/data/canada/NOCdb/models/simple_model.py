import pickle
from typing import Tuple, List
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ..readers import TitleSet, TitleRecord
from ..util import FeatureEngineer
from scribe_classifier.data.canada.NOCdb.readers.titles import TitlePreprocessor
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin


class SimpleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, target_level=1, emptyset_label: str=None, use_bayes=False):
        self.target_level = target_level
        self.use_bayes = use_bayes
        if use_bayes:
            # self.prop_records = 1.0/8.0
            self.parameters = {
                'vect__ngram_range': [(1, 2), (1, 3)],
                'clf__alpha': (1e-2, 1e-5)
            }
            self.clf_pipe = Pipeline([
                ('vect', CountVectorizer(stop_words='english')),
                ('clf', MultinomialNB(alpha=1e-4))
            ])
        else:
            # self.prop_records = 1.0
            self.parameters = {
                'vect__ngram_range': [(1, 2), (1, 3)],
                'clf__alpha': (1e-2, 1e-5),
                'clf__max_iter': (1000, 3000),
                'clf__tol': (1e-3, 1e-4)
            }
            self.clf_pipe = Pipeline([
                ('vect', CountVectorizer(stop_words='english')),
                ('clf', SGDClassifier(alpha=1e-4, max_iter=1000, tol=1e-4))
            ])

        self.clf = GridSearchCV(self.clf_pipe, self.parameters, n_jobs=-1)
        if emptyset_label is not None:
            if emptyset_label == "":
                self.emptyset_label = "NA"
            else:
                self.emptyset_label = emptyset_label

    def save_as_pickle(self, file, is_path=False):
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'SimpleModel':
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        smdl = pickle.load(handle)
        handle.close()
        return smdl

    def fit_titleset(self, title_set: 'TitleSet'):
        class_counts = title_set.count_classes()
        if self.emptyset_label is not None:
            # if not self.use_bayes:
            if self.use_bayes:
                prop_records = 0.25
            else:
                prop_records = 1.0 / float(len(class_counts))

            working_title_set = title_set.copy_and_append_empty_string_class(label=self.emptyset_label,
                                                                             prop_records=prop_records)
        else:
            working_title_set = title_set
        X, Y = working_title_set.split_into_title_and_code_vecs(target_level=self.target_level)
        self.clf.fit(X, Y)

    def predict_titleset(self, title_set: 'TitleSet') -> 'List[str]':
        return self.clf.predict(title_set.get_title_vec())

    def predict_titlevec(self, title_vec: 'List[TitleRecord]') -> 'List[str]':
        return self.clf.predict(title_vec)

    def predict_titlerecord(self, title_record: 'TitleRecord') -> str:
        print(type(title_record))
        return self.clf.predict([title_record.title])[0]

    def fit(self, X, y, **fit_params):
        tset = TitleSet()
        tset.add_titles_from_vecs(title_vec=X, code_vec=y)
        self.fit_titleset(title_set=tset)

    def predict(self, X):
        return self.predict_titlevec(X)



