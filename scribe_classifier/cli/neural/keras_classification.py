'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import click
from sklearn import metrics
from scribe_classifier.data.NOCdb.readers import TitleSet
from scribe_classifier.data.NOCdb.models.neural_networks import ANNclassifier
from scribe_classifier.data.NOCdb.models.ensemble.combined_models import CombinedModels


@click.group()
def keras_classifier_cli():
    pass


@keras_classifier_cli.command(name='train')
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
@click.option('--warmstart/--no-warmstart', default=False, help="continue training existing model")
def keras_classifier_train(target_level, model_filepath, epoch, layer, activation, max_features, first_layer_size, batch_size, warmstart, train_filepath, test_filepath):
    train = TitleSet.load_from_pickle(file=train_filepath)
    test = TitleSet.load_from_pickle(file=test_filepath)
    x_train = train.get_title_vec()
    x_test = test.get_title_vec()
    y_train = train.get_code_vec(target_level=target_level)
    y_test = test.get_code_vec(target_level=target_level)

    if warmstart:
        print("Loading Existing Model")
        mdl = ANNclassifier.load_from_pickle(model_filepath)
        print(mdl.model.summary())
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


@keras_classifier_cli.command(name='test')
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
@click.option('--model_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="Location where model in pickle format is located")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="Location where training set will be read in pickle format")
@click.option('--test_filepath', type=click.File('rb'), required=True, help="Location where test set will be read in pickle format")
def test_model(target_level, model_filepath, train_filepath, test_filepath):
    train = TitleSet.load_from_pickle(file=train_filepath)
    test = TitleSet.load_from_pickle(file=test_filepath)
    x_train = train.get_title_vec()
    x_test = test.get_title_vec()
    y_train = train.get_code_vec(target_level=target_level)
    y_test = test.get_code_vec(target_level=target_level)

    mdl = ANNclassifier.load_from_pickle(model_filepath)
    print(mdl.model.summary())
    mdl.set_warm_start(state=False)
    mdl.evaluation_metrics(x_test=x_test, y_test=y_test, x_valid=x_train, y_valid=y_train)


@keras_classifier_cli.command(name='freeze')
@click.option('--model', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="Location where model in pickle format is located")
@click.option('--frozen', type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True), required=True, help="Location where model in pickle format is located")
def freeze_copy(model, frozen):
    mdl = ANNclassifier.load_from_pickle(model)
    mdl.save_as_pickle(filepath=frozen, include_optimizer=False)


@keras_classifier_cli.command(name='combined_test')
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format. It will be used to understand how to stratify the models")
@click.option('--model',
              type=click.Tuple((click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
                    click.INT,
                    click.STRING)),
              multiple=True,
              help="provide a model to use, its type [sgd, bayes, ann], and specify its target level")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--val', type=click.File('rb'), required=True)
@click.option('--test', type=click.File('rb'), required=True)
@click.option('--lowmem/--no-lowmem', default=False, help="Randomly Sample 20% of validation set (use if memory errors are a problem)")
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def combine_test(target_level, model, emptyset, val, test, code_file, lowmem):
    if emptyset == "":
        emptyset = "NA"
    mdl_paths = dict()
    for tup in model:
        if tup[1] not in mdl_paths:
            mdl_paths[tup[1]] = dict()
        mdl_paths[tup[1]][tup[2]] = tup[0]
    if 1 not in mdl_paths:
        mdl_paths[1] = None
    if 2 not in mdl_paths:
        mdl_paths[2] = None
    if 3 not in mdl_paths:
        mdl_paths[3] = None
    cmb_mdls = CombinedModels(lvl1_mdls=mdl_paths[1],
                              lvl2_mdls=mdl_paths[2],
                              lvl3_mdls=mdl_paths[3],
                              target_level=target_level,
                              all_codes=code_file,
                              emptyset_label=emptyset)
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
    valid_p = cmb_mdls.predict(validset.get_title_vec())
    test_p = cmb_mdls.predict(testset.get_title_vec())

    print("Validation Set:")
    print(metrics.classification_report(valid_y, valid_p))
    print("Test Set:")
    print(metrics.classification_report(test_y, test_p))

    print("Val  Acc: ", metrics.accuracy_score(valid_y, valid_p),
          "Test Acc", metrics.accuracy_score(test_y, test_p))

if __name__ == "__main__":
    keras_classifier_cli()
