import pandas as pd
import psycopg2
from flask import render_template
from flask import request
from sqlalchemy import create_engine
from canada_model import *
from canada_data.combine_strings import *
from flask_demo import app
from flask_demo.a_Model import ModelIt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scribe_data.dbhandler import DataFramePickler

user = 'mgooch' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'scribe'
db = create_engine('postgres://%s@%s/%s'%(user, host, dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)


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

        cats = np.ndarray.copy(self.cats)  # type: np.ndarray
        cats = np.append(cats, ["Avg / Total"])
        df['Category'] = cats
        df.index = pd.RangeIndex(len(df.index))
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        return df


model = SimpleModel.load_from_pickle('./TrainedModels/simple.P', is_path=True)  # type: SimpleModel
valid = DataSet.load_from_pickle('./Validation_And_Test_Sets/simple.valid.set.P', is_path=True)  # type: DataSet
test = DataSet.load_from_pickle('./Validation_And_Test_Sets/simple.test.set.P', is_path=True)  # type: DataSet
valid_pred = model.clf.predict(valid.X)
test_pred = model.clf.predict(test.X)

valid_report = ClassificationReporter(valid.Y, valid_pred, valid.encoder.classes_)
test_report = ClassificationReporter(test.Y, test_pred, test.encoder.classes_)

combined_model = SimpleModel.load_from_pickle('./TrainedModels/simple.combined.P', is_path=True)  # type: SimpleModel
combined_valid = DataSet.load_from_pickle('./Validation_And_Test_Sets/simple.valid.set.combined.P', is_path=True)  # type: DataSet
combined_test = DataSet.load_from_pickle('./Validation_And_Test_Sets/simple.test.set.combined.P', is_path=True)  # type: DataSet
combined_valid_pred = combined_model.clf.predict(combined_valid.X)
combined_test_pred = combined_model.clf.predict(combined_test.X)

combined_valid_report = ClassificationReporter(combined_valid.Y, combined_valid_pred, combined_valid.encoder.classes_)
combined_test_report = ClassificationReporter(combined_test.Y, combined_test_pred, combined_test.encoder.classes_)

code_table = read_levels('./TrainingData/training_sources/raw/NOC/all_codes')
code_table['NA'] = "Not able to classify"

scribe_query_df = DataFramePickler.load_from_pickle('./SavedScribeQueries/midsize_tech_usa.P')


def do_scribe_predicts(combined: bool, label='class'):
    titles = scribe_query_df['title']
    titles.fillna(value="", inplace=True)
    # print(titles)
    if combined:
        titles_pred = combined_model.clf.predict(titles)
    else:
        titles_pred = model.clf.predict(titles)
    # print(titles_pred)
    scribe_query_df[label] = pd.Series(titles_pred)


do_scribe_predicts(False, 'class')
do_scribe_predicts(True, 'combined_class')


def generate_canada_category_plot(output_fname, empty_class):
    code_file = './TrainingData/training_sources/raw/NOC/all_codes'
    example_file = './TrainingData/training_sources/raw/NOC/all_examples'
    dataset = DataSet.from_files(code_file, example_file, 2, False, empty_class)
    # print(dataset.encoder.classes_)
    df = pd.DataFrame()
    df['description'] = pd.Series(dataset.X)
    df['class'] = pd.Series(dataset.Y)
    # print(dataset.Y)
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 9)
    sns.countplot(data=df, x='class', ax=ax)
    fig.savefig(output_fname)


def generate_scribe_category_plot(output_fname, label: str ='class'):
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 9)
    scribe_query_df.sort_values(label, inplace=True)
    # print(scribe_query_df[label])
    sns.countplot(data=scribe_query_df, x=label, ax=ax)
    fig.savefig(output_fname)
    pass


@app.route('/')
@app.route('/index')
def index():
    img_path = os.path.abspath('./flask_demo/static/img/canada_histogram.png')
    img_path_with_emptycat = os.path.abspath('./flask_demo/static/img/canada_histogram_emptycat.png')
    force = True
    if force or not os.path.exists(img_path):
        generate_canada_category_plot(img_path, False)
    if force or not os.path.exists(img_path_with_emptycat):
        generate_canada_category_plot(img_path_with_emptycat, True)
    return render_template("index.html",title = 'Home', user = { 'nickname': 'Miguel' },)


@app.route('/model_uncombined_validate')
def uncombined_validation_results_page():
    return render_template('model_validate.html',
                           switch_link_url="/model_uncombined_test",
                           switch_link_text="Uncombined Test Set Classification Metrics",
                           model_type="Validation Set: Uncombined Job Title Text",
                           dataframe=valid_report.get_report_dataframe().to_html(index=False))


@app.route('/model_combined_validate')
def combined_validation_results_page():
    return render_template('model_validate.html',
                           switch_link_url="/model_combined_test",
                           switch_link_text="Combined Test Set Classification Metrics",
                           model_type="Validation Set: Category Descriptions Combined With Job Title Text",
                           dataframe=combined_valid_report.get_report_dataframe().to_html(index=False))


@app.route('/model_uncombined_test')
def uncombined_test_results_page():
    return render_template('model_validate.html',
                           switch_link_url="/model_uncombined_validate",
                           switch_link_text="Uncombined Validation Set Classification Metrics",
                           model_type="Test Set: Uncombined Job Title Text",
                           dataframe=test_report.get_report_dataframe().to_html(index=False))


@app.route('/model_combined_test')
def combined_test_results_page():
    return render_template('model_validate.html',
                           switch_link_url="model_combined_validate",
                           switch_link_text="Combined Validation Set Classification Metrics",
                           model_type="Test Set: Category Descriptions Combined With Job Title Text ",
                           dataframe=combined_test_report.get_report_dataframe().to_html(index=False))


@app.route('/input')
def classify_text_input():
    return render_template("input.html")


@app.route('/future_plans')
def future_plans():
    return render_template("future_plans.html")


@app.route('/scribe_results')
def scribe_results():
    #Index(['id', 'email', 'firstName', 'lastName', 'company', 'industry', 'title',
    # 'companyCity', 'companyCountry', 'employeeCount', 'emailError', 'emailvalidity']
    query_string ="select * from email_list where \"companyCountry\" = 'United States' and \"industry\" in ('computer software','information technology and services,internet','marketing and advertising','internet') and \"employeeCount\" < 500 and \"emailError\" = False;"
    img_path = os.path.abspath('./flask_demo/static/img/usa_midsize_tech_histogram.png')
    combined_img_path = os.path.abspath('./flask_demo/static/img/combined_usa_midsize_tech_histogram.png')
    force = True
    if force or not os.path.exists(img_path):
        generate_scribe_category_plot(img_path, 'class')
    if force or not os.path.exists(combined_img_path):
        generate_scribe_category_plot(combined_img_path, 'combined_class')
    return render_template("scribe_results.html", query_string=query_string)


@app.route('/output', methods=['POST'])
def classify_text_output():
    test_text = request.form['job_title_test']
    test_pred = model.clf.predict([test_text])
    pred_descript = code_table[test_pred[0]]
    # print(test_pred[0])
    return render_template("output.html", test_text=test_text, class_id=test_pred[0], class_text=pred_descript)