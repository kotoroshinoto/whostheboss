#!/usr/bin/env python
import click
from sklearn import metrics
from scribe_classifier.data.canada import AllCodes, TitleSet, SimpleModel, MultiStepModel


@click.group()
def canada_model_cli():
    """Tools for working with Canadian JobTitle Database"""
    pass


@canada_model_cli.command(name="gen_data")
@click.option('--target', type=click.IntRange(1, 4), default=2, help="train against this code abstraction level")
@click.option('--example_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format")
@click.option('--train_filepath', type=click.File('wb'), default=None, help="Location where training set will be saved in pickle format")
@click.option('--valid_filepath', type=click.File('wb'), default=None, help="Location where validation set will be saved in pickle format")
@click.option('--test_filepath', type=click.File('wb'), default=None, help="Location where test set will be saved in pickle format")
@click.option('--valid_prop', type=click.FLOAT, default=0.20, help="value between 0.0 and 1.0 designation proportion to be used for validation set")
@click.option('--test_prop', type=click.FLOAT, default= 0.20, help="value between 0.0 and 1.0 designation proportion to be used for test set")
def generate_data_set(target, example_file, train_filepath, valid_filepath, test_filepath, valid_prop, test_prop):
    """Generate Train/Validate/Test Sets"""
    all_titles = TitleSet()
    # print("Reading Titles")
    all_titles.add_titles_from_file(filename=example_file)
    # print("Reading Titles Complete:")
    counts = all_titles.count_classes()
    # for cat in counts:
    #     print("%s\t%d" % (cat, counts[cat]))
    old_len = len(all_titles.records)
    # print("NA\t%d" % (len(all_titles.records) - old_len))
    train, valid, test = all_titles.split_data_valid_train_test(test_split=test_prop, valid_split=valid_prop, target_level=target)
    if train_filepath is None:
        train_filepath=open('./pickles/Validation_And_Test_Sets/train.set.P', 'wb')
    if valid_filepath is None:
        valid_filepath = open('./pickles/Validation_And_Test_Sets/valid.set.P', 'wb')
    if test_filepath is None:
        test_filepath = open('./pickles/Validation_And_Test_Sets/test.set.P', 'wb')
    train.save_as_pickle(file=train_filepath, is_path=False)
    valid.save_as_pickle(file=valid_filepath, is_path=False)
    test.save_as_pickle(file=test_filepath, is_path=False)


@canada_model_cli.command()
@click.option('--model_filepath', type=click.File('wb'), default=None, help="Location where model will be saved in pickle format")
@click.option('--train_filepath', type=click.File('rb'), default=None, help="Location where training set will be read in pickle format")
@click.option('--target_level', type=click.IntRange(1, 4), default=1, help="train against this code abstraction level")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to training set before fitting, if you provide an empty string label, default 'NA' will be used instead")
def simple(model_filepath, train_filepath, target_level, emptyset):
    """Use Simple Model, predict one specific category level all at once"""
    train = TitleSet.load_from_pickle(train_filepath)
    mdl = SimpleModel(target_level=target_level, emptyset_label=emptyset)
    mdl.fit(title_set=train)
    if model_filepath is None:
        model_filepath=open('./pickles/TrainedModels/simple.P', 'wb')
    mdl.save_as_pickle(file=model_filepath, is_path=False)


@canada_model_cli.command()
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.argument('model_file', type=click.File('rb'), required=True)
@click.argument('validation_file', type=click.File('rb'), required=True)
@click.argument('test_file', type=click.File('rb'), required=True)
@click.argument('target', type=click.IntRange(1, 4), default=1)
def test_simple(emptyset, model_file, validation_file, test_file, target):
    """Run test on model with validation set and test set"""
    model = SimpleModel.load_from_pickle(model_file)  # type: SimpleModel
    valid = TitleSet.load_from_pickle(validation_file)  # type: TitleSet
    test = TitleSet.load_from_pickle(test_file)  # type: TitleSet

    if emptyset is not None:
        valid = valid.copy_and_append_empty_string_class(label=emptyset)
        test = test.copy_and_append_empty_string_class(label=emptyset)

    valid_pred = model.predict(valid)
    valid_target_codes = valid.get_code_vec(target_level=target)

    test_pred = model.predict(test)
    test_target_codes = test.get_code_vec(target_level=target)

    print("Validation Set:")
    print(metrics.classification_report(valid_target_codes, valid_pred))
    # print(metrics.confusion_matrix(valid.Y, valid_pred))

    print("Test Set:")
    print(metrics.classification_report(test_target_codes, test_pred))
    # print(metrics.confusion_matrix(test.Y, test_pred))


@canada_model_cli.command(name='multi')
@click.option('--code_file', type=click.File('r'), required=True, help="This file should contain all codes and descriptions in tab-separated format")
@click.option('--example_file', type=click.File('r'), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format")
@click.option('--model_filepath', type=click.File('wb'), default='./TrainedModels/multi.P', help="Location where model will be saved in pickle format")
def multi_step(code_file, example_file, model_filepath):
    """Use Multi-Step Model, Predicts one layer of granularity at a time\n
    It will train multiple sub-models to discriminate among the subclasses of the upper category level"""
    raise click.BadArgumentUsage("multi command currently a placeholder for later development")


if __name__ == "__main__":
    canada_model_cli()
