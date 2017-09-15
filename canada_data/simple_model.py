import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from canada_data.dataset import DataSet


class SimpleModel:
    def __init__(self):
        self.parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              # 'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
        self.clf_pipe = Pipeline([('vect', CountVectorizer(stop_words='english')),
                 # ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB(alpha=1e-3))])
        self.clf = GridSearchCV(self.clf_pipe, self.parameters, n_jobs=-1)

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
        handle.close
        return smdl

    @classmethod
    def new_from_files(cls, code_file, example_file, target_level=2, combine=False, append_empty_class=False, test_split=0.20, valid_split=0.20) -> 'Tuple[SimpleModel, DataSet, DataSet]':
        """Will return the SimpleModel, a validation set, and a test set in a tuple"""
        dataset = DataSet.from_files(code_file, example_file, target_level, combine, append_empty_class)
        # cat_counts = dataset.count_classes()
        # for cat in cat_counts:
        #     print("%s\t%d" % (cat, cat_counts[cat]))
        train, valid, test = dataset.split_data_valid_train_test(test_split=test_split, valid_split=valid_split)

        simple_mdl = cls()
        simple_mdl.clf.fit(train.X, train.Y)
        return simple_mdl, valid, test


