import io

import pandas as pd
from flask import render_template
from flask import request

from scribe_classifier.data.NOCdb.models.ensemble.combined_models import CombinedModels
from scribe_classifier.data.NOCdb.models.neural_networks import ANNclassifier
from scribe_classifier.data.NOCdb.models.simple import SimpleModel
from scribe_classifier.data.NOCdb.readers.codes import CodeSet
from scribe_classifier.flask_demo import app
from scribe_classifier.flaskprep import ClassificationReporter

pd.set_option('display.max_colwidth', -1)

all_codes = CodeSet.load_from_pickle('./scribedata/source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
all_codes.add_emptyset()
classes = all_codes.get_codes_for_level(target_level=2)

mdl_paths = dict()
for target_level in range(1, 5):
    level_mdl_paths = dict()
    level_mdl_paths['sgd'] = SimpleModel.load_from_pickle('scribedata/source_data/pickles/canada/trained_models/simple.lvl%d.sgdsv.P' % target_level, is_path=True)
    level_mdl_paths['bayes'] = SimpleModel.load_from_pickle('scribedata/source_data/pickles/canada/trained_models/simple.lvl%d.bayes.P' % target_level, is_path=True)
    level_mdl_paths['neural'] = ANNclassifier.load_from_pickle('scribedata/nnmodels/ANN/neural_net_level%d.P' % target_level)
    mdl_paths[target_level] = level_mdl_paths

#models
models = CombinedModels(all_codes='source_data/pickles/canada/tidy_sets/all_codes.P')
models.add_models_from_dict(mdl_paths)

valid_report = ClassificationReporter.load_from_pickle(filepath='scribe_classifier/flask_demo/pickles/report.valid.P')  # type: ClassificationReporter
test_report = ClassificationReporter.load_from_pickle(filepath='scribe_classifier/flask_demo/pickles/report.test.P')  # type: ClassificationReporter


@app.route('/')
@app.route('/index')
@app.route('/slides')
def slide_page():
    return render_template("slides.html")


@app.route('/classes')
def imbalanced_classes_page():
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
    return render_template("scribe_results.html", query_string=query_string)


@app.route('/input')
def classify_text_input():
    return render_template("input.html")


@app.route('/output', methods=['POST'])
def classify_text_output():
    test_text = request.form['job_title_test']
    test_pred1 = models.predict([test_text], target_level=1)[0]
    test_pred2 = models.predict([test_text], target_level=2)[0]
    test_pred3 = models.predict([test_text], target_level=3)[0]
    test_pred4 = models.predict([test_text], target_level=4)[0]
    pred_descript1 = all_codes.codes[test_pred1].desc
    pred_descript2 = all_codes.codes[test_pred2].desc
    pred_descript3 = all_codes.codes[test_pred3].desc
    pred_descript4 = all_codes.codes[test_pred4].desc
    # print(test_pred[0])
    return render_template("output.html",
                           test_text=test_text,
                           class_id1=test_pred1,
                           class_text1=pred_descript1,
                           class_id2=test_pred2,
                           class_text2=pred_descript2,
                           class_id3=test_pred3,
                           class_text3=pred_descript3,
                           class_id4=test_pred4,
                           class_text4=pred_descript4
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
    for i in range(1, 5):
        out_descs[i] = []
    for line in ss:  # type: str
        line = line.rstrip().lstrip()
        titles.append(line)
    df['Input'] = titles
    for i in range(1, 5):
        if models.calc_num_models(target_level=i) == 0.0:
            continue
        preds[i] = models.batched_predict(X=titles, target_level=i)
        for j in range(len(titles)):
            out_descs[i].append(all_codes.codes[preds[i][j]].desc)
        df['Level %d Code' % i] = preds[i]
        df['Level %d Description' % i] = out_descs[i]
    return render_template("multi_output.html", dataframe=df.to_html(index=False, classes=["table", "table-bordered"]))


