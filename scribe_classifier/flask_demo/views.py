import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scribe_classifier.data.canada.NOCdb.readers.codes import AllCodes, CodeRecord
from scribe_classifier.data.canada.NOCdb.readers.titles import TitleSet
from scribe_classifier.data.scribe import DataFramePickler
from flask import render_template
from flask import request
from sklearn import metrics

from scribe_classifier.data.canada.NOCdb.models import SimpleModel
from scribe_classifier.flask_demo import app


# user = 'mgooch' #add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'scribe'
# db = create_engine('postgres://%s@%s/%s'%(user, host, dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)


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


force_img_generation = False

scribe_query_df = DataFramePickler.load_from_pickle('./SavedScribeQueries/midsize_tech_usa.P')

all_codes = AllCodes.load_from_pickle('./source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
classes = all_codes.get_codes_for_level(target_level=2)
all_codes.add_code(CodeRecord(code="NA", desc="Not able to classify"))
classes.append("NA")

#models
simple_model = SimpleModel.load_from_pickle('./source_data/pickles/canada/trained_models/simple.lvl2.P', is_path=True)  # type: SimpleModel

#dataset
valid = TitleSet.load_from_pickle('./source_data/pickles/canada/test_sets/valid.set.lvl2.P', is_path=True)  # type: TitleSet
test = TitleSet.load_from_pickle('./source_data/pickles/canada/test_sets/test.set.lvl2.P', is_path=True)  # type: TitleSet



#predictions
valid_pred = simple_model.predict_titleset(valid)
test_pred = simple_model.predict_titleset(test)

#generate reports
valid_report = ClassificationReporter(valid.get_code_vec(target_level=2), valid_pred, classes=classes)
test_report = ClassificationReporter(test.get_code_vec(target_level=2), test_pred, classes=classes)


def do_scribe_predicts(label='class'):
    titles = scribe_query_df['title']
    titles.fillna(value="", inplace=True)
    # print(titles)
    titles_pred = simple_model.predict(titles)
    # print(titles_pred)
    scribe_query_df[label] = pd.Series(titles_pred)


do_scribe_predicts('class')
do_scribe_predicts('combined_class')


def generate_canada_category_plot(output_fname, add_empty_class):
    code_file = './TrainingData/training_sources/raw/NOC/all_codes'
    example_file = './TrainingData/training_sources/raw/NOC/all_examples'
    dataset = TitleSet()
    dataset.add_titles_from_file(filename=example_file)
    if add_empty_class:
        dataset.copy_and_append_empty_string_class()
    df = dataset.to_dataframe(target_level=2)
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
    if force_img_generation or not os.path.exists(img_path):
        generate_canada_category_plot(img_path, False)
    if force_img_generation or not os.path.exists(img_path_with_emptycat):
        generate_canada_category_plot(img_path_with_emptycat, True)
    return render_template("index.html", title ='Home', user = {'nickname': 'Miguel'}, )


@app.route('/model_uncombined_validate')
def uncombined_validation_results_page():
    return render_template('model_validate.html',
                           switch_link_url="/model_uncombined_test",
                           switch_link_text="Uncombined Test Set Classification Metrics",
                           model_type="Validation Set: Uncombined Job Title Text",
                           dataframe=valid_report.get_report_dataframe().to_html(index=False))


@app.route('/model_uncombined_test')
def uncombined_test_results_page():
    return render_template('model_validate.html',
                           switch_link_url="/model_uncombined_validate",
                           switch_link_text="Uncombined Validation Set Classification Metrics",
                           model_type="Test Set: Uncombined Job Title Text",
                           dataframe=test_report.get_report_dataframe().to_html(index=False))


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
    if force_img_generation or not os.path.exists(img_path):
        generate_scribe_category_plot(img_path, 'class')
    if force_img_generation or not os.path.exists(combined_img_path):
        generate_scribe_category_plot(combined_img_path, 'combined_class')
    return render_template("scribe_results.html", query_string=query_string)


@app.route('/output', methods=['POST'])
def classify_text_output():
    test_text = request.form['job_title_test']
    test_pred = simple_model.predict_titlerecord(test_text)
    code_record = all_codes.codes[test_pred[0]]  # type: CodeRecord
    pred_descript = code_record.desc
    # print(test_pred[0])
    return render_template("output.html", test_text=test_text, class_id=test_pred[0], class_text=pred_descript)