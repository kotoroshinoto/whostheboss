'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import click
from scribe_classifier.data.canada import TitleSet
from scribe_classifier.data.canada.NOCdb.models.neural_networks.artificial_neural_net import ANNclassifier


@click.group()
def keras_classifier_main():
    pass


@keras_classifier_main.command(name='train')
@click.argument('target_level', type=click.INT)
@click.option('--epoch', type=click.INT, default=10, help="# of epochs to use when training")
@click.option('--first_layer_size', type=click.INT, default=512, help="Size of Input Layer")
@click.option('--layer', type=(click.INT, click.INT, click.FLOAT), default=(512, 1, 0.0), multiple=True, help="triplet of values, # of neurons in layer, and # of layers, 3rd value is a bool, for whether to put a dropout layer after")
@click.option('--activation', type=click.STRING, default='sigmoid', help="neuron activation types (except last layer) https://keras.io/activations/")
@click.option('--max_features', type=click.INT, default=10000, help="max features from count vectorizer")
@click.option('--batch_size', type=click.INT, default=64, help="batch size for tensorflow")
def keras_classifier_train(target_level, epoch, layer, activation, max_features, first_layer_size, batch_size):
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
    # print(y_train)
    # print(y_test)
    mdl = ANNclassifier(target_level=target_level,
                        epochs=epoch,
                        max_words=max_features,
                        layer_def=layer,
                        first_layer_size=first_layer_size,
                        batch_size=batch_size)
    mdl.fit(x=x_train, y=y_train, validation_data=(x_test, y_test))

    mdl.evaluation_metrics(x_test=x_test, y_test=y_test, x_valid=x_train, y_valid=y_train)

    mdl.save_as_pickle('/home/mgooch/PycharmProjects/insight/nnmodels/neural_net_level%d.P' % target_level)


if __name__ == "__main__":
    keras_classifier_main()
