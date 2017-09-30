import pickle
import pandas as pd
from sklearn import metrics


class ClassificationReporter:
    def __init__(self, y, y_pred, classes):
        retvals =[]
        for retval in metrics.precision_recall_fscore_support(y, y_pred, average='weighted'):
            retvals.append(retval)
        self.avg_precision = retvals[0]
        self.avg_recall = retvals[1]
        self.avg_fbeta_score = retvals[2]
        self.total = len(y)
        retvals = []
        for retval in metrics.precision_recall_fscore_support(y, y_pred):
            retvals.append(retval)
        self.precision = retvals[0]
        self.recall = retvals[1]
        self.fbeta_score = retvals[2]
        self.support = retvals[3]
        self.conf_matrix = metrics.confusion_matrix(y, y_pred)
        self.cats = classes

    def get_report_dataframe(self):
        df = pd.DataFrame()
        df['Precision'] = pd.Series(data=self.precision).append(pd.Series([self.avg_precision]))
        df['Recall'] = pd.Series(data=self.recall).append(pd.Series([self.avg_recall]))
        df['F1-Score'] = pd.Series(data=self.fbeta_score).append(pd.Series([self.avg_fbeta_score]))
        df['Support'] = pd.Series(data=self.support).append(pd.Series([self.total]))
        cats = list(self.cats)
        cats.append("Avg / Total")
        df['Category'] = cats
        df.index = pd.RangeIndex(len(df.index))
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        return df

    def save_as_pickle(self, filepath):
        pickle_file = open(filepath, 'wb')
        pickle.dump(self, pickle_file)

    @staticmethod
    def load_from_pickle(filepath):
        pickle_file = open(filepath, 'rb')
        return pickle.load(pickle_file)


class ObjectPickler:
    @staticmethod
    def save_as_pickle(obj, filepath):
        pickle_file = open(filepath, 'wb')
        pickle.dump(obj, pickle_file)

    @staticmethod
    def load_from_pickle(filepath):
        pickle_file = open(filepath,'rb')
        return pickle.load(pickle_file)
