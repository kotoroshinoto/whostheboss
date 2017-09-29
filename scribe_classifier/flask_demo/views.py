import io
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flask import render_template
from flask import request

from preconstruct_eval_dataframes_for_flask import ClassificationReporter, scribe_query_df, do_scribe_predicts
from scribe_classifier.data.canada.NOCdb.models.neural_networks.combined_models import CombinedModels
from scribe_classifier.data.canada.NOCdb.readers.codes import AllCodes, CodeRecord
from scribe_classifier.data.canada.NOCdb.readers.titles import TitleSet
from scribe_classifier.flask_demo import app

# user = 'mgooch' #add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'scribe'
# db = create_engine('postgres://%s@%s/%s'%(user, host, dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

pd.set_option('display.max_colwidth', -1)

force_img_generation = False

all_codes = AllCodes.load_from_pickle('./source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
classes = all_codes.get_codes_for_level(target_level=2)
all_codes.add_code(CodeRecord(code="NA", desc="Not able to classify"))
classes.append("NA")

mdl_strs = dict()
models = dict()  # type: Dict[int, CombinedModels]
for target_level in range(1, 4):
    level_mdl_strs = dict()
    level_mdl_strs['sgd'] = 'source_data/pickles/canada/trained_models/simple.lvl%d.sgdsv.P' % target_level
    level_mdl_strs['bayes'] = 'source_data/pickles/canada/trained_models/simple.lvl%d.bayes.P' % target_level
    level_mdl_strs['ann'] = 'nnmodels/ANN/neural_net_level%d.frozen.P' % target_level
    mdl_strs[target_level] = level_mdl_strs


#models
for target_level in range(1, 4):
    try:
        models[target_level] = CombinedModels('source_data/pickles/canada/tidy_sets/all_codes.P',
                                              mdl_strs[1],
                                              mdl_strs[2],
                                              mdl_strs[3],
                                              target_level=target_level)
    except MemoryError:
        print("Ran out of memory loading combined models")

#dataset
valid = TitleSet.load_from_pickle('./source_data/pickles/canada/test_sets/train.set.lvl4.P', is_path=True)  # type: TitleSet
test = TitleSet.load_from_pickle('./source_data/pickles/canada/test_sets/test.set.lvl4.P', is_path=True)  # type: TitleSet
train, valid = valid.split_data_train_test(target_level=4, test_split=0.25)
del train
train = None
valid = valid.copy_and_append_empty_string_class("NA")
test = test.copy_and_append_empty_string_class("NA")

print("# validation records: ", len(valid.records))
print('# test records: ', len(test.records))

#predictions
valid_pred = []
test_pred = []
try:
    valid_pred = models[2].batched_predict(valid.get_title_vec())
    test_pred = models[2].batched_predict(test.get_title_vec())
except MemoryError:
    print("had memory error trying to predict on records")

#generate reports
valid_report = ClassificationReporter(valid.get_code_vec(target_level=2), valid_pred, classes=classes)
test_report = ClassificationReporter(test.get_code_vec(target_level=2), test_pred, classes=classes)

do_scribe_predicts('class')


def generate_canada_category_plot(output_fname, add_empty_class):
    code_file = './source_data/pickles/canada/tidy_sets/all_codes.P'
    example_file = './source_data/pickles/canada/tidy_sets/all_titles.P'
    dataset = TitleSet()
    dataset =TitleSet.load_from_pickle(filename=example_file)
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
@app.route('/slides')
def slide_page():
    return render_template("slides.html")


@app.route('/classes')
def imbalanced_classes_page():
    img_path = os.path.abspath('./scribe_classifier/flask_demo/static/img/canada_histogram.png')
    img_path_with_emptycat = os.path.abspath('./scribe_classifier/flask_demo/static/img/canada_histogram_emptycat.png')
    if force_img_generation or not os.path.exists(img_path):
        generate_canada_category_plot(img_path, False)
    if force_img_generation or not os.path.exists(img_path_with_emptycat):
        generate_canada_category_plot(img_path_with_emptycat, True)
    return render_template("imbalanced_classes.html")


@app.route('/model_validate')
def validation_results_page():
    return render_template('model_validate.html',
                           model_type="Validation Set:",
                           dataframe=valid_report.get_report_dataframe().to_html(index=False, classes=["table", "table-bordered", "table-striped"]))


@app.route('/model_test')
def test_results_page():
    return render_template('model_validate.html',
                           model_type="Test Set:",
                           dataframe=test_report.get_report_dataframe().to_html(index=False, classes=["table", "table-bordered", "table-striped"]))


@app.route('/future_plans')
def future_plans():
    return render_template("future_plans.html")


@app.route('/scribe_results')
def scribe_results():
    #Index(['id', 'email', 'firstName', 'lastName', 'company', 'industry', 'title',
    # 'companyCity', 'companyCountry', 'employeeCount', 'emailError', 'emailvalidity']
    query_string ="select * from email_list where \"companyCountry\" = 'United States' and \"industry\" in ('computer software','information technology and services,internet','marketing and advertising','internet') and \"employeeCount\" < 500 and \"emailError\" = False;"
    img_path = os.path.abspath('./scribe_classifier/flask_demo/static/img/usa_midsize_tech_histogram.png')
    combined_img_path = os.path.abspath('./scribe_classifier/flask_demo/static/img/combined_usa_midsize_tech_histogram.png')
    if force_img_generation or not os.path.exists(img_path):
        generate_scribe_category_plot(img_path, 'class')
    if force_img_generation or not os.path.exists(combined_img_path):
        generate_scribe_category_plot(combined_img_path, 'combined_class')
    return render_template("scribe_results.html", query_string=query_string)


@app.route('/input')
def classify_text_input():
    return render_template("input.html")


@app.route('/output', methods=['POST'])
def classify_text_output():
    test_text = request.form['job_title_test']
    test_pred1 = models[1].predict([test_text])[0]
    test_pred2 = models[2].predict([test_text])[0]
    test_pred3 = models[3].predict([test_text])[0]
    pred_descript1 = all_codes.codes[test_pred1].desc
    pred_descript2 = all_codes.codes[test_pred2].desc
    pred_descript3 = all_codes.codes[test_pred3].desc
    # print(test_pred[0])
    return render_template("output.html",
                           test_text=test_text,
                           class_id1=test_pred1,
                           class_text1=pred_descript1,
                           class_id2=test_pred2,
                           class_text2=pred_descript2,
                           class_id3=test_pred3,
                           class_text3=pred_descript3
                           )


@app.route('/multi_input')
def classify_text_input_multi():
    return render_template("multi_input.html")


@app.route('/multi_output', methods=['POST'])
def classify_text_output_multi():
    test_text = request.form['job_title_test']
    ss = io.StringIO(test_text)
    df = pd.DataFrame()
    titles = []
    out_descs = dict()
    preds = dict()
    for i in range(1, 4):
        out_descs[i] = []
    for line in ss:  # type: str
        line = line.rstrip().lstrip()
        titles.append(line)
    try:
        preds[1] = models[1].predict(titles)
        preds[2] = models[2].predict(titles)
        preds[3] = models[3].predict(titles)

    except MemoryError:
        df = pd.DataFrame()
        df['ERROR'] = ["Ran out of memory, try using fewer inputs"]
        return render_template("multi_output.html", dataframe=df.to_html(index=False, classes=["table", "table-bordered"]))

    for i in range(len(titles)):
        out_descs[1].append(all_codes.codes[preds[1][i]].desc)
        out_descs[2].append(all_codes.codes[preds[2][i]].desc)
        out_descs[3].append(all_codes.codes[preds[3][i]].desc)

    df['Input'] = titles
    df['Level1_Code'] = preds[1]
    df['Level1_Description'] = out_descs[1]
    df['Level2_Code'] = preds[2]
    df['Level2_Description'] = out_descs[2]
    df['Level3_Code'] = preds[3]
    df['Level3_Description'] = out_descs[3]
    return render_template("multi_output.html", dataframe=df.to_html(index=False, classes=["table", "table-bordered"]))


