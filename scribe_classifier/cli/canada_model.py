#!/usr/bin/env python
import click
from sklearn import metrics
from scribe_classifier.data.canada import AllCodes, TitleSet, SimpleModel, MultiStepModel


@click.group()
def canada_model_cli():
    """Tools for working with Canadian JobTitle Database"""
    pass


@canada_model_cli.group(name ='dataset')
def gen_data_set():
    """generate training and test set"""
    pass


def split_and_pickle_dataset(tset, target_level, train_filepath, valid_filepath, test_filepath, test_prop, valid_prop):
    train, valid, test = tset.split_data_valid_train_test(test_split=test_prop, valid_split=valid_prop, target_level=target_level)
    if train_filepath is None:
        train_filepath=open('./source_data/pickles/canada/test_sets/train.set.P', 'wb')
    if valid_filepath is None:
        valid_filepath = open('./source_data/pickles/canada/test_sets/valid.set.P', 'wb')
    if test_filepath is None:
        test_filepath = open('./source_data/pickles/canada/test_sets/test.set.P', 'wb')
    train.save_as_pickle(file=train_filepath, is_path=False)
    valid.save_as_pickle(file=valid_filepath, is_path=False)
    test.save_as_pickle(file=test_filepath, is_path=False)


@gen_data_set.command(name="file")
@click.option('--target_level', type=click.IntRange(1, 4), default=2, help="train against this code abstraction level")
@click.option('--example_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format")
@click.option('--train_filepath', type=click.File('wb'), default=None, help="Location where training set will be saved in pickle format")
@click.option('--valid_filepath', type=click.File('wb'), default=None, help="Location where validation set will be saved in pickle format")
@click.option('--test_filepath', type=click.File('wb'), default=None, help="Location where test set will be saved in pickle format")
@click.option('--valid_prop', type=click.FLOAT, default=0.20, help="value between 0.0 and 1.0 designation proportion to be used for validation set")
@click.option('--test_prop', type=click.FLOAT, default= 0.20, help="value between 0.0 and 1.0 designation proportion to be used for test set")
def generate_data_set(target_level, example_file, train_filepath, valid_filepath, test_filepath, valid_prop, test_prop):
    """Generate Train/Validate/Test Sets"""
    all_titles = TitleSet()
    all_titles.add_titles_from_file(filename=example_file)
    split_and_pickle_dataset(
        tset=all_titles,
        target_level=target_level,
        train_filepath=train_filepath,
        valid_filepath=valid_filepath,
        test_filepath=test_filepath,
        valid_prop=valid_prop,
        test_prop=test_prop
    )


@gen_data_set.command(name="pickle")
@click.option('--target_level', type=click.IntRange(1, 4), default=2, help="train against this code abstraction level")
@click.option('--example_file', type=click.File('rb'), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format, pre-loaded into a TitleSet in pickle format")
@click.option('--train_filepath', type=click.File('wb'), default=None, help="Location where training set will be saved in pickle format")
@click.option('--valid_filepath', type=click.File('wb'), default=None, help="Location where validation set will be saved in pickle format")
@click.option('--test_filepath', type=click.File('wb'), default=None, help="Location where test set will be saved in pickle format")
@click.option('--valid_prop', type=click.FLOAT, default=0.20, help="value between 0.0 and 1.0 designation proportion to be used for validation set")
@click.option('--test_prop', type=click.FLOAT, default= 0.20, help="value between 0.0 and 1.0 designation proportion to be used for test set")
def generate_data_set_from_pickle(target_level, example_file, train_filepath, valid_filepath, test_filepath, valid_prop, test_prop):
    all_titles = TitleSet.load_from_pickle(file=example_file, is_path=False)
    split_and_pickle_dataset(
        tset=all_titles,
        target_level=target_level,
        train_filepath=train_filepath,
        valid_filepath=valid_filepath,
        test_filepath=test_filepath,
        valid_prop=valid_prop,
        test_prop=test_prop
    )


@canada_model_cli.group()
def simple():
    """Work with Simple Models"""
    pass


@simple.command(name="train")
@click.option('--model_filepath', type=click.File('wb'), required=True, help="Location where model will be saved in pickle format")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="Location where training set will be read in pickle format")
@click.option('--target_level', type=click.IntRange(1, 4), default=1, help="train against this code abstraction level")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to training set before fitting, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--oversample/--no-oversample', default=False, help="toggle random oversampling, will use MultinomialNB model in this mode instead of SGDClassifier")
@click.option('--bayes/--no-bayes', default=False, help='Use MultinomialNB instead of SGDClassifier')
def generate_simple_model(model_filepath, train_filepath, target_level, emptyset, oversample, bayes):
    """Use Simple Model, predict one specific category level all at once, using SGDClassifier"""
    train = TitleSet.load_from_pickle(train_filepath)
    if oversample or bayes:
        mdl = SimpleModel(target_level=target_level, emptyset_label=emptyset, use_bayes=True)
    else:
        mdl = SimpleModel(target_level=target_level, emptyset_label=emptyset)
    if oversample:
        # print("oversampling")
        train = train.copy_and_oversample_to_flatten_stratification()
    mdl.fit_titleset(title_set=train)
    if model_filepath is None:
        model_filepath=open('./pickles/TrainedModels/simple.P', 'wb')
    mdl.save_as_pickle(file=model_filepath, is_path=False)


@simple.command(name="test")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.argument('model_file', type=click.File('rb'), required=True)
@click.argument('validation_file', type=click.File('rb'), required=True)
@click.argument('test_file', type=click.File('rb'), required=True)
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def test_simple_model(emptyset, model_file, validation_file, test_file, target_level):
    """Run test on model with validation set and test set"""
    model = SimpleModel.load_from_pickle(model_file)  # type: SimpleModel
    valid = TitleSet.load_from_pickle(validation_file)  # type: TitleSet
    test = TitleSet.load_from_pickle(test_file)  # type: TitleSet

    if emptyset is not None:
        valid = valid.copy_and_append_empty_string_class(label=emptyset)
        test = test.copy_and_append_empty_string_class(label=emptyset)

    valid_pred = model.predict_titleset(valid)
    valid_target_codes = valid.get_code_vec(target_level=target_level)

    test_pred = model.predict_titleset(test)
    test_target_codes = test.get_code_vec(target_level=target_level)

    print("Validation Set:")
    print(metrics.classification_report(valid_target_codes, valid_pred))
    # print(metrics.confusion_matrix(valid.Y, valid_pred))

    print("Test Set:")
    print(metrics.classification_report(test_target_codes, test_pred))
    # print(metrics.confusion_matrix(test.Y, test_pred))


@canada_model_cli.group()
def multi():
    """Work with Multi Step Models"""
    pass


@multi.command(name='train')
@click.option('--target_level', type=click.IntRange(1, 4), default=1, help="train against this code abstraction level")
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False,readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format. It will be used to understand how to stratify the models")
@click.option('--model_filepath', type=click.File('wb'), required=True, help="Location where model will be saved in pickle format")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="Location where training set will be read in pickle format")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to training set before fitting, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--oversample/--no-oversample', default=False, help="toggle random oversampling")
def generate_multi_step_model(code_file, train_filepath, model_filepath, emptyset, target_level, oversample):
    """Use Multi-Step Model, Predicts one layer of granularity at a time\n
    It will train multiple sub-models to discriminate among the subclasses of the upper category level"""
    train_titleset = TitleSet.load_from_pickle(file=train_filepath, is_path=False)
    if oversample:
        print("oversampling")
        train_titleset = train_titleset.copy_and_oversample_to_flatten_stratification()
    msm = MultiStepModel(target_level=target_level, all_codes_filename=code_file, emptyset_label=emptyset)
    msm.fit(title_set=train_titleset)
    msm.save_as_pickle(model_filepath, is_path=False)


@multi.command(name="test")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.argument('model_file', type=click.File('rb'), required=True)
@click.argument('validation_file', type=click.File('rb'), required=True)
@click.argument('test_file', type=click.File('rb'), required=True)
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def test_multi_step_model(emptyset, model_file, validation_file, test_file, target_level):
    model = MultiStepModel.load_from_pickle(model_file)  # type: MultiStepModel
    valid = TitleSet.load_from_pickle(validation_file)  # type: TitleSet
    test = TitleSet.load_from_pickle(test_file)  # type: TitleSet

    if emptyset is not None:
        valid = valid.copy_and_append_empty_string_class(label=emptyset)
        test = test.copy_and_append_empty_string_class(label=emptyset)

    valid_pred = model.predict(valid)
    test_pred = model.predict(test)

    for i in range(1, target_level+1):
        valid_target_codes = valid.get_code_vec(target_level=i)
        test_target_codes = test.get_code_vec(target_level=i)
        valid_level_pred = valid_pred.get_preds(target_level=i)
        test_level_pred = test_pred.get_preds(target_level=i)
        print("Validation Set:")
        print(metrics.classification_report(valid_target_codes, valid_level_pred))
        # print(metrics.confusion_matrix(valid.Y, valid_pred))

        print("Test Set:")
        print(metrics.classification_report(test_target_codes, test_level_pred))
        # print(metrics.confusion_matrix(test.Y, test_pred))


if __name__ == "__main__":
    canada_model_cli()
