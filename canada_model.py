#!/usr/bin/env python
import pickle
import click
from sklearn import metrics
from canada_data.dataset import DataSet
from canada_data.simple_model import SimpleModel


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
@click.option('--emptyset/--no-emptyset', default=False, help="Add Empty String Dataset labeled 'Unknown'")
def simple(code_file, example_file, model_filepath, valid_filepath, test_filepath, target, combine, valid_prop, test_prop, emptyset):
    """Use Simple Model, predict one specific category level all at once"""
    mdl, valid, test = SimpleModel.new_from_files(code_file=code_file,
                                                  example_file=example_file,
                                                  target_level=target,
                                                  combine=combine,
                                                  valid_split=valid_prop,
                                                  test_split=test_prop,
                                                  append_empty_class=emptyset)
    if model_filepath is None:
        model_filepath=open('./TrainedModels/simple.P', 'wb')
    if valid_filepath is None:
        valid_filepath = open('./Validation_And_Test_Sets/simple.valid.set.P', 'wb')
    if test_filepath is None:
        test_filepath = open('./Validation_And_Test_Sets/simple.test.set.P', 'wb')
    mdl.save_as_pickle(model_filepath)
    valid.save_as_pickle(valid_filepath)
    test.save_as_pickle(test_filepath)


@main.command()
@click.argument('model_file', type=click.File('rb'), required=True)
@click.argument('validation_file', type=click.File('rb'), required=True)
@click.argument('test_file', type=click.File('rb'), required=True)
def test_simple(model_file, validation_file, test_file):
    """Run test on model with validation set and test set"""
    model = SimpleModel.load_from_pickle(model_file)  # type: SimpleModel
    valid = DataSet.load_from_pickle(validation_file)  # type: DataSet
    valid_pred = model.clf.predict(valid.X)
    print("Validation Set:")
    print(metrics.classification_report(valid.Y, valid_pred))
    # print(metrics.confusion_matrix(valid.Y, valid_pred))
    test = DataSet.load_from_pickle(test_file)  # type: DataSet
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