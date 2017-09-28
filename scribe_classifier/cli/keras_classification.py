'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import click
from scribe_classifier.data.canada import TitleSet
from scribe_classifier.data.canada.NOCdb.models.neural_networks.artificial_neural_net import ANNclassifier
from sklearn import metrics


ann_filepath = '/home/mgooch/PycharmProjects/insight/nnmodels/ANN/neural_net_level%d.P'


@click.group()
def keras_classifier_cli():
    pass


def load_sets(target_level: int):
    # print(type(layer))
    train = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/train.set.lvl%d.P' % target_level,
                                      is_path=True).copy_and_append_empty_string_class()
    test = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/test.set.lvl%d.P' % target_level,
                                     is_path=True).copy_and_append_empty_string_class()
    # valid = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/valid.set.lvl%d.P' % target_level,
    #                                   is_path=True).copy_and_append_empty_string_class()
    # counts = train.count_classes(target_level=4)
    # print(counts)
    # exit()

    x_train = train.get_title_vec()
    y_train = train.get_code_vec(target_level=target_level)
    # x_valid = valid.get_title_vec()
    # y_valid = valid.get_code_vec(target_level=target_level)
    x_test = test.get_title_vec()
    y_test = test.get_code_vec(target_level=target_level)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    return x_train, y_train, x_test, y_test


@keras_classifier_cli.command(name='train')
@click.argument('target_level', type=click.IntRange(1,4), default=1)
@click.option('--epoch', type=click.INT, default=10, help="# of epochs to use when training")
@click.option('--first_layer_size', type=click.INT, default=512, help="Size of Input Layer")
@click.option('--layer', type=(click.INT, click.INT, click.FLOAT), default=(512, 1, 0.0), multiple=True, help="triplet of values, # of neurons in layer, and # of layers, 3rd value is a bool, for whether to put a dropout layer after")
@click.option('--activation', type=click.STRING, default='sigmoid', help="neuron activation types (except last layer) https://keras.io/activations/")
@click.option('--max_features', type=click.INT, default=10000, help="max features from count vectorizer")
@click.option('--batch_size', type=click.INT, default=64, help="batch size for tensorflow")
@click.option('--warmstart/--no-warmstart', default=False, help="continue training existing model")
def keras_classifier_train(target_level, epoch, layer, activation, max_features, first_layer_size, batch_size, warmstart):
    x_train, y_train, x_test, y_test = load_sets(target_level=target_level)
    # print(y_train)
    # print(y_test)
    if warmstart:
        print("Loading Existing Model")
        mdl = ANNclassifier.load_from_pickle(ann_filepath % target_level)
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

    mdl.save_as_pickle(ann_filepath % target_level)


@keras_classifier_cli.command(name='test')
@click.argument('target_level', type=click.IntRange(1, 4), default=1)
def test_model(target_level):
    x_train, y_train, x_test, y_test = load_sets(target_level=target_level)
    mdl = ANNclassifier.load_from_pickle(ann_filepath % target_level)
    print(mdl.model.summary())
    mdl.set_warm_start(state=False)
    mdl.evaluation_metrics(x_test=x_test, y_test=y_test, x_valid=x_train, y_valid=y_train)
    test_pred = mdl.predict(x_test)
    train_pred = mdl.predict(x_train)


if __name__ == "__main__":
    keras_classifier_cli()
