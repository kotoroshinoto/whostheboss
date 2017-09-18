import pickle
from typing import Tuple
from ..readers import TitleSet, TitleRecord
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class SimpleModel:
    def __init__(self, target_level=1, emptyset_label: str=None):
        self.target_level = target_level
        self.parameters = {
            'vect__ngram_range': [(1, 1), (2, 4)],
            'clf__alpha': (1e-2, 1e-5),
            'clf__max_iter': (1000, 3000),
            'clf__tol': (1e-3, 1e-4)
        }

        self.clf_pipe = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('clf', SGDClassifier(alpha=1e-4, max_iter=1000, tol=1e-4))
            # ('clf', MultinomialNB(alpha=1e-3))
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

    def fit(self, title_set: 'TitleSet'):
        if self.emptyset_label is not None:
            working_title_set = title_set.copy_and_append_empty_string_class(label=self.emptyset_label)
        else:
            working_title_set = title_set
        X, Y = working_title_set.split_into_title_and_code_vecs(target_level=self.target_level)
        self.clf.fit(X, Y)

    def predict(self, title_set: 'TitleSet'):
        return self.clf.predict(title_set.get_title_vec())

    def predict_one(self, title_record: 'TitleRecord') -> str:
        return self.clf.predict([title_record.title])[0]

    @classmethod
    def new_from_files(cls, example_file, target_level=2, append_empty_class=False,
                       test_split=0.20, valid_split=0.20) -> 'Tuple[SimpleModel, TitleSet, TitleSet]':
        """Will return the SimpleModel, a validation set, and a test set in a tuple"""
        dataset = TitleSet()
        dataset.add_titles_from_file(filename=example_file)
        if append_empty_class:
            dataset.copy_and_append_empty_string_class()
        train, valid, test = dataset.split_data_valid_train_test(test_split=test_split, valid_split=valid_split)
        train_x, train_y = train.split_into_title_and_code_vecs(target_level=target_level)
        simple_mdl = cls()
        simple_mdl.clf.fit(train_x, train_y)
        return simple_mdl, valid, test


