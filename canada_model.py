#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from canada_data.combine_strings import *
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import click
from typing import List,Dict,Tuple
import pickle


def append_empty_string_class(xvec, yvec, label='Unknown'):
    num_to_add = int(len(xvec)/4)
    for i in range(num_to_add):
        xvec.append("")
        yvec.append(label)


class DataSet:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def count_classes(self)->'Dict[str, int]':
        counts = dict()
        for cat in self.Y:
            if cat not in counts:
                counts[cat] = 1
            else:
                counts[cat] += 1
        return counts

    @classmethod
    def from_vecs_split_data_valid_train_test(cls, xvec, yvec, test_split=0.20, valid_split=0.20) -> 'Tuple[DataSet, DataSet, DataSet]':
        return cls(xvec,yvec).split_data_valid_train_test(test_split=test_split, valid_split=valid_split)

    def split_data_valid_train_test(self, test_split=0.20, valid_split=0.20) -> 'Tuple[DataSet, DataSet, DataSet]':
        # split datasets into train/validation/test
        x_train, x_split, y_train, y_split = train_test_split(self.X, self.Y,
                                                              stratify=self.Y,
                                                              test_size=(test_split + valid_split))

        x_valid, x_test, y_valid, y_test = train_test_split(x_split, y_split,
                                                            stratify=y_split,
                                                            test_size=(test_split / (test_split + valid_split)))
        # put validation set back into training set
        x_train += x_valid
        y_train += y_valid

        # save these for return values
        train = self.__class__(X=x_train, Y=y_train)
        valid = self.__class__(X=x_valid, Y=y_valid)
        test = self.__class__(X=x_test, Y=y_test)
        return train, valid, test

    @classmethod
    def from_files(cls, code_file, example_file, target_level, combine=False, append_empty_class=False):
        level_codes = read_levels(code_file)
        title_records = read_titles(example_file, target_level)
        if combine:
            (xvec, yvec) = generate_combined(level_codes, title_records, target_level)
        else:
            (xvec, yvec) = generate_uncombined_text_for_target_level(title_records, target_level)

        if append_empty_class:
            append_empty_string_class(xvec, yvec)
        return cls(xvec, yvec)


class SimpleModel:
    def __init__(self):
        self.parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
        self.clf_pipe = Pipeline([('vect', CountVectorizer(stop_words='english')),
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB(alpha=1e-3))])
        self.clf = GridSearchCV(self.clf_pipe, self.parameters, n_jobs=-1)

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


# predict 0-9 & Unknown @ first level, then internal classes at next level
class MultiStepModel:
    def __init__(self):
        pass


@click.group()
def main():
    pass


@main.command()
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format")
@click.option('--example_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format")
@click.option('--model_filepath', type=click.File('wb'), default=None, help="Location where model will be saved in pickle format")
@click.option('--valid_filepath', type=click.File('wb'), default=None, help="Location where validation set will be saved in pickle format")
@click.option('--test_filepath', type=click.File('wb'), default=None, help="Location where test set will be saved in pickle format")
@click.option('--target', type=click.IntRange(1, 4), default=2, help="train against this code abstraction level")
@click.option('--combine/--no-combine', default=False)
@click.option('--valid_prop', type=click.FLOAT, default=0.20, help="value between 0.0 and 1.0 designation proportion to be used for validation set")
@click.option('--test_prop', type=click.FLOAT, default= 0.20, help="value between 0.0 and 1.0 designation proportion to be used for test set")
def simple(code_file, example_file, model_filepath, valid_filepath, test_filepath, target, combine, valid_prop, test_prop):
    """Use Simple Model, predict one specific category level all at once"""
    mdl, valid, test = SimpleModel.new_from_files(code_file=code_file,
                                                  example_file=example_file,
                                                  target_level=target,
                                                  combine=combine,
                                                  valid_split=valid_prop,
                                                  test_split=test_prop)
    if model_filepath is None:
        model_filepath=open('./TrainedModels/simple.P', 'wb')
    if valid_filepath is None:
        valid_filepath = open('./Validation_And_Test_Sets/simple.valid.set.P', 'wb')
    if test_filepath is None:
        test_filepath = open('./Validation_And_Test_Sets/simple.test.set.P', 'wb')

    pickle.dump(mdl, model_filepath)
    pickle.dump(valid, valid_filepath)
    pickle.dump(test, test_filepath)


@main.command()
@click.argument('model_file', type=click.File('rb'), required=True)
@click.argument('validation_file', type=click.File('rb'), required=True)
@click.argument('test_file', type=click.File('rb'), required=True)
def test_simple(model_file, validation_file, test_file):
    """Run test on model with validation set and test set"""
    model = pickle.load(model_file)  # type: SimpleModel
    valid = pickle.load(validation_file)  # type: DataSet
    valid_pred = model.clf.predict(valid.X)
    print("Validation Set:")
    print(metrics.classification_report(valid.Y, valid_pred))
    # print(metrics.confusion_matrix(valid.Y, valid_pred))
    test = pickle.load(test_file)  # type: DataSet
    test_pred = model.clf.predict(test.X)
    print("Test Set:")
    print(metrics.classification_report(test.Y, test_pred))
    # print(metrics.confusion_matrix(test.Y, test_pred))


@main.command(name='multi')
@click.option('--code_file', type=click.File('r'), required=True, help="This file should contain all codes and descriptions in tab-separated format")
@click.option('--example_file', type=click.File('r'), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format")
@click.option('--model_filepath', type=click.File('wb'), default='./TrainedModels/multi.P', help="Location where model will be saved in pickle format")
def multi_step(code_file, example_file, model_filepath):
    """Use Multi-Step Model, Predicts one layer of granularity at a time\n
    It will train multiple sub-models to discriminate among the subclasses of the upper category level"""
    raise NotImplementedError("Currently a placeholder for later development")


if __name__ == "__main__":
    main()