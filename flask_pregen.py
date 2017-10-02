#!/usr/bin/env python
import os
import click
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scribe_classifier.data.NOCdb.models.ensemble import CombinedModels
from scribe_classifier.data.NOCdb.readers import TitleSet, CodeSet
from scribe_classifier.data.scribe import DataFramePickler
from scribe_classifier.flaskprep import ClassificationReporter


def get_combined_models():
    """assembles a combined models object with all available models"""
    mdl_strs = dict()
    for target_level in range(1, 5):
        level_mdl_strs = dict()
        level_mdl_strs['sgd'] = 'source_data/pickles/canada/trained_models/simple.lvl%d.sgdsv.P' % target_level
        level_mdl_strs['bayes'] = 'source_data/pickles/canada/trained_models/simple.lvl%d.bayes.P' % target_level
        level_mdl_strs['neural'] = 'nnmodels/ANN/neural_net_level%d.P' % target_level
        mdl_strs[target_level] = level_mdl_strs
    # models
    models = CombinedModels('source_data/pickles/canada/tidy_sets/all_codes.P',
                            mdl_strs[1],
                            mdl_strs[2],
                            mdl_strs[3],
                            mdl_strs[4]
                            )
    return models


def classify_scribe_data(scribe_query_df, batch_size, keras_batch_size=4000, label='class'):
    """Classifies scribe's data into level 2 buckets and adds a new column with the labels"""
    models = get_combined_models()
    titles = scribe_query_df['title']
    titles.fillna(value="", inplace=True)
    # print("# of titles in scribe db: ", len(titles))
    titles_pred = models.batched_predict(titles, batch_size=batch_size, keras_batch_size=keras_batch_size, target_level=2)
    # print(titles_pred)
    scribe_query_df[label] = pd.Series(titles_pred)


def classify_test_set(batch_size, keras_batch_size=4000):
    """Classifies test set and validation set, and returns the resulting reports and prediction vectors"""
    models = get_combined_models()
    all_codes = CodeSet.load_from_pickle('./source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
    all_codes.add_emptyset()
    classes = all_codes.get_codes_for_level(target_level=2)
    # dataset
    train = TitleSet.load_from_pickle('./source_data/pickles/canada/test_sets/train.set.lvl4.P', is_path=True)  # type: TitleSet
    test = TitleSet.load_from_pickle('./source_data/pickles/canada/test_sets/test.set.lvl4.P', is_path=True)  # type: TitleSet
    train, valid = train.split_data_train_test(target_level=4, test_split=0.25)
    del train
    train = None
    valid = valid.copy_and_append_empty_string_class("NA")
    test = test.copy_and_append_empty_string_class("NA")
    print("# validation records: ", len(valid.records))
    print('# test records: ', len(test.records))
    # predictions
    valid_pred = []
    test_pred = []
    try:
        valid_pred = models.batched_predict(valid.get_title_vec(), batch_size=batch_size, keras_batch_size=keras_batch_size, target_level=2)
        test_pred = models.batched_predict(test.get_title_vec(), batch_size=batch_size, keras_batch_size=keras_batch_size, target_level=2)
    except MemoryError:
        print("had memory error trying to predict on records")
    # generate reports
    valid_report = ClassificationReporter(valid.get_code_vec(target_level=2), valid_pred, classes=classes)
    test_report = ClassificationReporter(test.get_code_vec(target_level=2), test_pred, classes=classes)
    return valid_report, test_report, valid_pred, test_pred


@click.group()
def main_flask_prep():
    """Prepare images and dataframes for the flask app"""
    pass


@main_flask_prep.command(name='reports')
@click.option('--batch_size', type=click.INT, default=4000, help='set number of records to classify at once')
@click.option('--keras_batch_size', type=click.INT, default=4000, help='set keras batch size parameter')
def generate_canada_reports(batch_size, keras_batch_size):
    """create dataframes for html reports on classification metrics"""
    valid_report, test_report, valid_pred, test_pred = classify_test_set(
        batch_size=batch_size,
        keras_batch_size=keras_batch_size
    )
    valid_report.save_as_pickle(filepath="scribe_classifier/flask_demo/pickles/report.valid.P")
    test_report.save_as_pickle(filepath="scribe_classifier/flask_demo/pickles/report.test.P")


@main_flask_prep.command(name='scribe_df')
@click.option('--batch_size', type=click.INT, default=4000, help='set number of records to classify at once')
@click.option('--keras_batch_size', type=click.INT, default=4000, help='set keras batch size parameter')
def scribe_dataframe(batch_size, keras_batch_size):
    """Classify Scribe data and save as a pickle"""
    scribe_query_df = DataFramePickler.load_from_pickle('./SavedScribeQueries/midsize_tech_usa.P')
    classify_scribe_data(scribe_query_df,
                         label='class',
                         batch_size=batch_size,
                         keras_batch_size=keras_batch_size
                         )
    DataFramePickler.save_as_pickle(df=scribe_query_df,
                                    filepath='SavedScribeQueries/classified/scribe_classified_df.P')


def generate_canada_category_plot(output_fname, add_empty_class, target_level=2):
    """Generates plots of the number of titles within each category in the canadian NOC database at target level"""
    code_file = './source_data/pickles/canada/tidy_sets/all_codes.P'
    ac = CodeSet.load_from_pickle(file=code_file, is_path=True)
    example_file = './source_data/pickles/canada/tidy_sets/all_titles.P'
    dataset = TitleSet.load_from_pickle(file=example_file, is_path=True)
    if add_empty_class:
        dataset = dataset.copy_and_append_empty_string_class()
    df = dataset.to_dataframe(target_level=target_level)
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 9)
    sns.countplot(data=df, x='codes', ax=ax)
    fig.savefig(output_fname)


@main_flask_prep.command(name='canada_plots')
@click.option('--force/--no-force', default=False, help='Re-generate images even if they already exist')
def generate_canada_plots(force):
    """generate some plots from the dataframe and train/test set pickles"""
    print("Creating Metrics plots")
    canada_img_path = os.path.abspath('./scribe_classifier/flask_demo/static/img/canada_histogram.png')
    canada_img_path_with_emptycat = os.path.abspath('./scribe_classifier/flask_demo/static/img/canada_histogram_emptycat.png')
    if force or not os.path.exists(canada_img_path):
        generate_canada_category_plot(canada_img_path, False)
    if force or not os.path.exists(canada_img_path_with_emptycat):
        generate_canada_category_plot(canada_img_path_with_emptycat, True)


def generate_scribe_category_plot(scribe_query_df, output_fname, label: str ='class'):
    """Generates plots of the number of items with label in dataframe"""
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 9)
    scribe_query_df.sort_values(label, inplace=True)
    # print(scribe_query_df[label])
    sns.countplot(data=scribe_query_df, x=label, ax=ax)
    fig.savefig(output_fname)
    pass


@main_flask_prep.command(name='scribe_plots')
@click.option('--force/--no-force', default=False, help='Re-generate images even if they already exist')
def generate_scribe_plots(force):
    print("Creating Scribe Plot")
    scribe_query_df = DataFramePickler.load_from_pickle('./SavedScribeQueries/classified/scribe_classified_df.P')
    scribe_img_path = os.path.abspath('./scribe_classifier/flask_demo/static/img/usa_midsize_tech_histogram.png')
    if force or not os.path.exists(scribe_img_path):
        generate_scribe_category_plot(scribe_query_df, scribe_img_path, 'class')


if __name__ == "__main__":
    main_flask_prep()
