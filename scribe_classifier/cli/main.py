#!/usr/bin/env python
import sys
import os
import click
from sklearn import metrics
from scribe_classifier.data.NOCdb.readers import TitleRecord, TitleSet, CodeRecord, CodeSet, TitlePreprocessor
from scribe_classifier.data.NOCdb.models.ensemble_util.ensemble_funcs import predict_from_files
from scribe_classifier.data.NOCdb.readers import CodeSet, TitleSet


@click.group()
def canada_model_cli():
    """Tools for working with models based on the Canadian NOC JobTitle Database"""
    pass


@canada_model_cli.group()
def NOCdb():
    """Operations to clean and prepare NOC database for use"""
    pass


@NOCdb.command(name='codes')
@click.option('--infile', '-i', type=click.File('r', encoding='iso-8859-1'), required=True, help="original raw original NOC code file")
@click.option('--outfile', '-o', type=click.File('w'), default=None, help="file to write mapped output to in tabular format")
@click.option('--pickle', '-p', type=click.File('wb'), default=None, help="file to write mapped output to in pickle format")
def clean_codes(infile, outfile, pickle):
    from csv import reader
    """This program cleans up the NOC code file, saving the resulting code set in a pickle format.\n
    User can redirect stdout if they desire to save a tabular copy of the same data."""
    if outfile is None:
        outfile = sys.stdout
    rdr = reader(infile)
    first_line = next(rdr)
    # print("\t".join([first_line[0], first_line[1], first_line[3], first_line[5]]))
    filestr = 'code_lvl_%d'
    all_codes = CodeSet()
    for entry in rdr:
        if entry[1].rstrip().lstrip() == "":
            continue
        code_l = CodeSet.parse_code_column(entry[0])
        for code_s in code_l:
            kept_values = [code_s, entry[1], entry[3], entry[5]]
            if pickle is not None:
                all_codes.add_code(CodeRecord(code=code_s, desc=entry[1]))
            print("\t".join(kept_values), file=outfile)
    if pickle is not None:
        all_codes.save_as_pickle(file=pickle, is_path=False)


@NOCdb.command(name='titles')
@click.option('--infile', '-i', type=click.File('r', encoding='iso-8859-1'), required=True, help="original raw original NOC example file")
@click.option('--outfile', '-o', type=click.File('w'), default=None, help="file to write mapped output to in tabular format")
@click.option('--pickle', '-p', type=click.File('wb'), default=None, help="file to write mapped output to in pickle format")
def clean_titles(infile, outfile, pickle):
    from csv import reader
    """This program will clean NOC title examples, splitting dual-gender titles, condensing some prefixes
    and rips out non alphanumeric characters, leaving single whitespaces between words.\n
    It creates a pickle file containing the prepared titleset. \n
    User can redirect stdout to capture a tabular copy of the file as well"""
    if outfile is None:
        outfile = sys.stdout
    rdr = reader(infile)
    first_line = next(rdr)
    tset = TitleSet()
    for entry in rdr:
        if int(entry[3]) != 19:
            continue
        if entry[1].rstrip().lstrip() == "":
            continue
        kept_values = [entry[0], entry[1], entry[3], entry[4]]
        print("\t".join(kept_values), file=outfile)
        if pickle is not None:
            tset.add_title(TitleRecord(code=entry[0], title=entry[1]))

    if pickle is not None:
        tset = TitlePreprocessor.preprocess_titleset_split_genders(tset=tset)
        tset = TitlePreprocessor.preprocess_titleset_split_genders(tset=tset)
        # tset = TitlePreprocessor.preprocess_titleset_split_chief_officer(tset=tset)
        tset = TitlePreprocessor.preprocess_titleset_prefixes(tset=tset)
        tset = TitlePreprocessor.preprocess_slugify_titleset(tset=tset)
        for trecord in tset.records:
            print("%s\t%s" % (trecord.code, trecord.title))
        tset.save_as_pickle(file=pickle, is_path=False)


@canada_model_cli.group(name ='dataset')
def gen_data_set():
    """generate training and test set"""
    pass


def split_and_pickle_dataset(tset, target_level, train_filepath, valid_filepath, test_filepath, test_prop, valid_prop):
    """This function splits a dataset and pickles the results"""
    if valid_prop == 0.0:
        train, test = tset.split_data_train_test(target_level=target_level, test_split=test_prop)
    else:
        train, valid, test = tset.split_data_valid_train_test(test_split=test_prop, valid_split=valid_prop, target_level=target_level)
        if valid_filepath is None:
            valid_filepath = open('./source_data/pickles/canada/test_sets/valid.set.P', 'wb')
        valid.save_as_pickle(file=valid_filepath, is_path=False)

    if train_filepath is None:
        train_filepath=open('./source_data/pickles/canada/test_sets/train.set.P', 'wb')
    if test_filepath is None:
        test_filepath = open('./source_data/pickles/canada/test_sets/test.set.P', 'wb')
    train.save_as_pickle(file=train_filepath, is_path=False)
    test.save_as_pickle(file=test_filepath, is_path=False)


@gen_data_set.command(name="file")
@click.option('--target_level', type=click.IntRange(1, 4), default=4, help="int between 1 and 4 specifying the code level to use for stratified splits")
@click.option('--example_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all examples with their level 4 coding in tab-separated format")
@click.option('--train_filepath', type=click.File('wb'), default=None, help="Location where training set will be saved in pickle format")
@click.option('--valid_filepath', type=click.File('wb'), default=None, help="Location where validation set will be saved in pickle format")
@click.option('--test_filepath', type=click.File('wb'), default=None, help="Location where test set will be saved in pickle format")
@click.option('--valid_prop', type=click.FLOAT, default=0.20, help="value between 0.0 and 1.0 designating proportion to be used for validation set")
@click.option('--test_prop', type=click.FLOAT, default=0.20, help="value between 0.0 and 1.0 designating proportion to be used for test set")
def generate_data_set(target_level, example_file, train_filepath, valid_filepath, test_filepath, valid_prop, test_prop):
    """Generate Train/Validate/Test Sets, from NOC text data"""
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
    """Generate Train/Validate/Test Sets, from NOC pickled data"""
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
def model():
    pass

@model.group()
def simple():
    """Work with simple Models"""
    pass


@simple.command(name="train")
@click.option('--model_filepath', type=click.File('wb'), required=True, help="Location where model will be saved in pickle format")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="Location where training set will be read in pickle format")
@click.option('--target_level', type=click.IntRange(1, 4), default=1, help="train against this code abstraction level")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to training set before fitting, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--oversample/--no-oversample', default=False, help="toggle random oversampling, will use MultinomialNB model in this mode instead of SGDClassifier")
@click.option('--model_type', type=click.STRING, default='sgdsv', help="specify model type, one of ['sgdsv', 'bayes', 'gauss']")
def generate_simple_model(model_filepath, train_filepath, target_level, emptyset, oversample, model_type):
    from scribe_classifier.data.NOCdb.models.simple import SimpleModel
    """Use simple Model, predict one specific category level all at once, using SGDClassifier, MultinomialBayes, or SVC"""
    train = TitleSet.load_from_pickle(train_filepath)
    if oversample:
        mdl = SimpleModel(target_level=target_level, emptyset_label=emptyset, model_type='bayes', oversampled=oversample)
    elif model_type == 'bayes':
        mdl = SimpleModel(target_level=target_level, emptyset_label=emptyset, model_type='bayes')
    else:
        mdl = SimpleModel(target_level=target_level, emptyset_label=emptyset, model_type=model_type)
    if oversample:
        # print("oversampling")
        train = train.copy_and_oversample_to_flatten_stratification()
    titles = train.get_title_vec()
    codes = train.get_code_vec(target_level=target_level)
    if oversample:
        mdl.initialize_vectorizer(X=titles, y=codes)
        for i in range(0, len(titles), 4000):
            end = min(len(titles), i+4000)
            print("training from %d to %d" % (i+1, end))
            titleslice = titles[i:end]
            codeslice = codes[i:end]
            mdl.partial_fit(X=titleslice, y=codeslice)
    else:
        mdl.fit_titleset(title_set=train)
    if model_filepath is None:
        if oversample:
            model_filepath = open('./pickles/TrainedModels/simple.lvl%d.%s.%s.P' % (target_level, 'oversample','bayes'), 'wb')
        else:
            model_filepath = open('./pickles/TrainedModels/simple.lvl%d.%s.P' % (target_level, model_type), 'wb')
    mdl.save_as_pickle(file=model_filepath, is_path=False)


@simple.command(name="test")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.argument('model_file', type=click.File('rb'), required=True)
@click.argument('validation_file', type=click.File('rb'), required=True)
@click.argument('test_file', type=click.File('rb'), required=True)
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def test_simple_model(emptyset, model_file, validation_file, test_file, target_level):
    from scribe_classifier.data.NOCdb.models.simple import SimpleModel
    """Run test on model with validation set and test set"""
    if emptyset == "":
        emptyset = "NA"
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
    print("Test Set:")
    print(metrics.classification_report(test_target_codes, test_pred))
    print("Val  Acc: ", metrics.accuracy_score(valid_target_codes, valid_pred),
          "Test Acc", metrics.accuracy_score(test_target_codes, test_pred))


@model.group()
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
    from scribe_classifier.data.NOCdb.models.multi_level import MultiStepModel
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
    from scribe_classifier.data.NOCdb.models.multi_level import MultiStepModel
    """Run test on multi-step model with validation set and test set"""
    if emptyset == "":
        emptyset = "NA"
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
        print("Test Set:")
        print(metrics.classification_report(test_target_codes, test_level_pred))
        print("Val  Acc: ", metrics.accuracy_score(valid_target_codes, valid_pred),
              "Test Acc", metrics.accuracy_score(test_target_codes, test_pred))


@model.group()
def neural_net():
    """work with keras neural network models"""
    pass


@neural_net.command(name='train')
@click.argument('target_level', type=click.IntRange(1,4), default=1)
@click.option('--model_filepath', type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True), required=True, help="Location where model in pickle format is located")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="Location where training set will be read in pickle format")
@click.option('--test_filepath', type=click.File('rb'), required=True, help="Location where test set will be read in pickle format")
@click.option('--epoch', type=click.INT, default=10, help="# of epochs to use when training")
@click.option('--first_layer_size', type=click.INT, default=512, help="Size of Input Layer")
@click.option('--layer', type=(click.INT, click.INT, click.FLOAT), default=(512, 1, 0.0), multiple=True, help="triplet of values, # of neurons in layer, and # of layers, 3rd value is a bool, for whether to put a dropout layer after")
@click.option('--activation', type=click.STRING, default='sigmoid', help="neuron activation types (except last layer) https://keras.io/activations/")
@click.option('--max_features', type=click.INT, default=10000, help="max features from count vectorizer")
@click.option('--batch_size', type=click.INT, default=64, help="batch size for tensorflow")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to training set before fitting, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--warmstart/--no-warmstart', default=False, help="continue training existing model")
def keras_classifier_train(target_level, model_filepath, epoch, layer, activation, max_features, first_layer_size, batch_size, warmstart, train_filepath, test_filepath, emptyset):
    from scribe_classifier.data.NOCdb.models.neural_networks import ANNclassifier
    """create and train a neural network using keras framework to classify job titles"""
    train = TitleSet.load_from_pickle(file=train_filepath)
    test = TitleSet.load_from_pickle(file=test_filepath)
    if emptyset is not None:
        train = train.copy_and_append_empty_string_class(label=emptyset)
        test = test.copy_and_append_empty_string_class(label=emptyset)
    x_train = train.get_title_vec()
    x_test = test.get_title_vec()
    y_train = train.get_code_vec(target_level=target_level)
    y_test = test.get_code_vec(target_level=target_level)
    if warmstart:
        print("Loading Existing Model")
        mdl = ANNclassifier.load_from_pickle(model_filepath)
        # print(mdl.model.summary())
    else:
        print("Assembling New Model")
        mdl = ANNclassifier(target_level=target_level,
                            epochs=epoch,
                            max_words=max_features,
                            layer_def=layer,
                            first_layer_size=first_layer_size,
                            batch_size=batch_size,
                            activation=activation)
    mdl.set_warm_start(state=warmstart)
    mdl.fit(x=x_train, y=y_train, validation_data=(x_test, y_test))

    mdl.evaluation_metrics(x_test=x_test, y_test=y_test, x_valid=x_train, y_valid=y_train)

    mdl.save_as_pickle(model_filepath)


@neural_net.command(name='test')
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
@click.option('--model_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="Location where model in pickle format is located")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="Location where training set will be read in pickle format")
@click.option('--test_filepath', type=click.File('rb'), required=True, help="Location where test set will be read in pickle format")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to training set before fitting, if you provide an empty string label, default 'NA' will be used instead")
def test_model(target_level, model_filepath, train_filepath, test_filepath, emptyset):
    from scribe_classifier.data.NOCdb.models.neural_networks import ANNclassifier
    """test keras neural network model with training set and test set"""
    train = TitleSet.load_from_pickle(file=train_filepath)
    test = TitleSet.load_from_pickle(file=test_filepath)
    if emptyset is not None:
        train = train.copy_and_append_empty_string_class(label=emptyset)
        test = test.copy_and_append_empty_string_class(label=emptyset)
    x_train = train.get_title_vec()
    x_test = test.get_title_vec()
    y_train = train.get_code_vec(target_level=target_level)
    y_test = test.get_code_vec(target_level=target_level)
    print("NA in y_test", 'NA' in y_test)

    mdl = ANNclassifier.load_from_pickle(model_filepath)
    print(mdl.model.summary())
    mdl.evaluation_metrics(x_test=x_test, y_test=y_test, x_valid=x_train, y_valid=y_train)


@model.group()
def ensemble():
    """predictions from ensemble probability matrices"""
    pass


@ensemble.command(name='test_predict_from_files')
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format. It will be used to understand how to stratify the models")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--val', type=click.File('rb'), required=True)
@click.option('--test', type=click.File('rb'), required=True)
@click.option('--basepath', default="./proba_matrix", type=click.Path(dir_okay=True, file_okay=False, resolve_path=True, writable=True))
@click.option('--model', required=True, type=(click.STRING, click.IntRange(1, 4)), multiple=True, help="modeltype and model's target level")
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def test_predict_from_files(code_file, emptyset, val, test, basepath, target_level, model):
    ac = CodeSet.load_from_pickle(code_file, is_path=True)
    ac.add_emptyset(emptyset_label=emptyset)
    include_dict = dict()
    for tup in model:
        if tup[1] not in include_dict:
            include_dict[tup[1]] = list()
        include_dict[tup[1]].append(tup[0])
    valid_tset = TitleSet.load_from_pickle(file=val, is_path=False).copy_and_append_empty_string_class(label=emptyset)
    test_tset = TitleSet.load_from_pickle(file=test, is_path=False).copy_and_append_empty_string_class(label=emptyset)
    valid_y = valid_tset.get_code_vec(target_level=target_level)
    test_y = test_tset.get_code_vec(target_level=target_level)
    valid_p = predict_from_files(
        basepath=basepath,
        prefix='valid',
        target_level=target_level,
        code_set=ac,
        emptyset_label=emptyset,
        include_dict=include_dict
    )

    test_p = predict_from_files(
        basepath=basepath,
        prefix='test',
        target_level=target_level,
        code_set=ac,
        emptyset_label=emptyset,
        include_dict=include_dict
    )
    print("Validation Set:")
    print(metrics.classification_report(valid_y, valid_p))
    print("Test Set:")
    print(metrics.classification_report(test_y, test_p))
    print("Val  Acc: ", metrics.accuracy_score(valid_y, valid_p),
          "Test Acc", metrics.accuracy_score(test_y, test_p))


@ensemble.command(name='test')
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format. It will be used to understand how to stratify the models")
@click.option('--model',
              type=click.Tuple((click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
                    click.INT,
                    click.STRING)),
              multiple=True,
              help="provide a model to use, its type [sgd, bayes, neural], and specify its target level")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--val', type=click.File('rb'), required=True)
@click.option('--test', type=click.File('rb'), required=True)
@click.option('--lowmem/--no-lowmem', default=False, help="Randomly Sample 20% of validation set (use if memory errors are a problem)")
@click.option('--batchsize', type=click.INT, default=4000, help="number of titles to predict per batch")
@click.option('--keras_batch', type=click.INT, default=32, help="keras batch size to use during prediction")
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def combine_test(target_level, model, emptyset, val, test, code_file, lowmem, batchsize, keras_batch):
    from scribe_classifier.data.NOCdb.models.ensemble.combined_models import CombinedModels
    from scribe_classifier.data.NOCdb.models.simple import SimpleModel
    from scribe_classifier.data.NOCdb.models.neural_networks import ANNclassifier
    """test a combined ensemble model with a validation and test set on a target level of classification"""
    if emptyset == "":
        emptyset = "NA"
    mdl_paths = dict()
    for tup in model:
        tlvl = tup[1]
        mdl_type = tup[2]  # type: str
        modelpath = tup[0]
        if tlvl not in mdl_paths:
            mdl_paths[tlvl] = dict()
        if mdl_type == 'ann':
            mdl_paths[tlvl][mdl_type] = ANNclassifier.load_from_pickle(modelpath)
        else:
            mdl_paths[tlvl][mdl_type] = SimpleModel.load_from_pickle(modelpath, is_path=True)
    if 1 not in mdl_paths:
        mdl_paths[1] = None
    if 2 not in mdl_paths:
        mdl_paths[2] = None
    if 3 not in mdl_paths:
        mdl_paths[3] = None
    if 4 not in mdl_paths:
        mdl_paths[4] = None
    cmb_mdls = CombinedModels(all_codes=code_file,
                              emptyset_label=emptyset)
    cmb_mdls.add_models_from_dict(mdl_paths)

    validset = TitleSet.load_from_pickle(val)  # type: TitleSet
    testset = TitleSet.load_from_pickle(test)  # type: TitleSet
    if emptyset is not None:
        validset = validset.copy_and_append_empty_string_class(label=emptyset)  # type: TitleSet
        testset = testset.copy_and_append_empty_string_class(label=emptyset)  # type: TitleSet
    if lowmem:
        train, validset = validset.split_data_train_test(target_level=4, test_split=0.25)
        # print(validset.get_code_vec(target_level=1))
    valid_y = validset.get_code_vec(target_level=target_level)
    test_y = testset.get_code_vec(target_level=target_level)
    valid_p = cmb_mdls.batched_predict(validset.get_title_vec(),
                                       batch_size=batchsize,
                                       target_level=target_level,
                                       keras_batch_size=keras_batch)
    test_p = cmb_mdls.batched_predict(testset.get_title_vec(),
                                      batch_size=batchsize,
                                      target_level=target_level,
                                      keras_batch_size=keras_batch)
    cmb_mdls.batched_predict_proba_per_model_to_files(X=testset.get_title_vec(),
                                                      batch_size=batchsize,
                                                      keras_batch_size=keras_batch,
                                                      path="./",
                                                      file_prefix="test.")
    cmb_mdls.batched_predict_proba_per_model_to_files(X=testset.get_title_vec(),
                                                      batch_size=batchsize,
                                                      keras_batch_size=keras_batch,
                                                      path="./",
                                                      file_prefix="valid.")
    print("Validation Set:")
    print(metrics.classification_report(valid_y, valid_p))
    print("Test Set:")
    print(metrics.classification_report(test_y, test_p))
    print("Val  Acc: ", metrics.accuracy_score(valid_y, valid_p),
          "Test Acc", metrics.accuracy_score(test_y, test_p))


@ensemble.command(name='test_to_files')
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format. It will be used to understand how to stratify the models")
@click.option('--model',
              type=click.Tuple((click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
                    click.INT,
                    click.STRING)),
              multiple=True,
              help="provide a model to use, its type [sgd, bayes, neural], and specify its target level")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--val', type=click.File('rb'), required=True)
@click.option('--test', type=click.File('rb'), required=True)
@click.option('--batchsize', type=click.INT, default=4000, help="number of titles to predict per batch")
@click.option('--keras_batch', type=click.INT, default=32, help="keras batch size to use during prediction")
@click.option('--basepath', default="./proba_matrix", type=click.Path(dir_okay=True, file_okay=False, resolve_path=True, writable=True))
def combine_test_files(model, emptyset, val, test, code_file, batchsize, keras_batch, basepath):
    from scribe_classifier.data.NOCdb.models.ensemble.combined_models import CombinedModels
    from scribe_classifier.data.NOCdb.models.simple import SimpleModel
    from scribe_classifier.data.NOCdb.models.neural_networks import ANNclassifier
    """test a combined ensemble model with a validation and test set on a target level of classification"""
    if emptyset == "":
        emptyset = "NA"
    mdl_paths = dict()
    for tup in model:
        tlvl = tup[1]
        mdl_type = tup[2]  # type: str
        modelpath = tup[0]
        if tlvl not in mdl_paths:
            mdl_paths[tlvl] = dict()
        if mdl_type == 'ann':
            mdl_paths[tlvl][mdl_type] = ANNclassifier.load_from_pickle(modelpath)
        else:
            mdl_paths[tlvl][mdl_type] = SimpleModel.load_from_pickle(modelpath, is_path=True)
    if 1 not in mdl_paths:
        mdl_paths[1] = None
    if 2 not in mdl_paths:
        mdl_paths[2] = None
    if 3 not in mdl_paths:
        mdl_paths[3] = None
    if 4 not in mdl_paths:
        mdl_paths[4] = None
    cmb_mdls = CombinedModels(all_codes=code_file,
                              emptyset_label=emptyset)
    cmb_mdls.add_models_from_dict(mdl_paths)
    validset = TitleSet.load_from_pickle(val)  # type: TitleSet
    testset = TitleSet.load_from_pickle(test)  # type: TitleSet
    if emptyset is not None:
        validset = validset.copy_and_append_empty_string_class(label=emptyset)  # type: TitleSet
        testset = testset.copy_and_append_empty_string_class(label=emptyset)  # type: TitleSet
    os.makedirs(basepath, exist_ok=True)
    cmb_mdls.batched_predict_proba_per_model_to_files(X=testset.get_title_vec(),
                                                      batch_size=batchsize,
                                                      keras_batch_size=keras_batch,
                                                      path=basepath,
                                                      file_prefix="test")
    cmb_mdls.batched_predict_proba_per_model_to_files(X=validset.get_title_vec(),
                                                      batch_size=batchsize,
                                                      keras_batch_size=keras_batch,
                                                      path=basepath,
                                                      file_prefix="valid")


if __name__ == "__main__":
    canada_model_cli()
