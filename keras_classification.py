'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from scribe_classifier.data.canada import TitleSet, TitleRecord
from scribe_classifier.data.canada import AllCodes, CodeRecord
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics import classification_report
from math import log2, pow
from keras.models import load_model
import pickle
import os
import os.path
import sys
import click


class ANNclassifier:
    def __init__(self, 
                 epochs=10,
                 target_level: int=1,
                 max_words: int=20000,
                 batch_size: int=64,
                 layer_def=((512, 1, True)),
                 first_layer_size=512):
        self.epochs = epochs
        self.target_level = target_level
        self.max_words = max_words
        self.batch_size = batch_size
        self.model = None  # type: Sequential
        self.history = None
        self.codes = None
        self.titles = None
        self.lbl_bin = None  # type: LabelBinarizer
        self.cvect = None  # type: CountVectorizer
        self.num_classes = None  # type: int
        self.layer_def = layer_def
        self.first_layer_size = first_layer_size

    def _load_assets(self):
        print('Loading data...')
        self.codes = AllCodes.load_from_pickle('source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
        self.codes.add_emptyset()
        self.titles = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P', is_path=True).copy_and_append_empty_string_class()

    def _setup_for_fit(self):
        self._load_assets()
        self.cvect = CountVectorizer(ngram_range=(1, 6), stop_words='english', max_features=self.max_words)
        self.lbl_bin = LabelBinarizer()
        ac_vec = self.codes.get_codes_for_level(target_level=self.target_level)
        print("classes: ", ac_vec)
        all_texts = self.titles.get_title_vec()
        all_title_codes = self.titles.get_code_vec(target_level=self.target_level)
        self.lbl_bin.fit(ac_vec)
        classes = self.lbl_bin.transform(ac_vec)
        self.num_classes = len(ac_vec)
        print(self.num_classes, 'classes')
        self.cvect.fit(all_texts, y=all_title_codes)
        self._assemble_model()

    def fit(self, x, y, validation_data: 'Tuple[List[str], List[str]]'=None):
        self._setup_for_fit()
        print('Vectorizing sequence data...')
        X = self.cvect.transform(x).todense()
        vX = self.cvect.transform(validation_data[0]).todense()
        print('x shape: ', X.shape, " type: ", type(x))
        Y = self.lbl_bin.transform(y)
        vY = self.lbl_bin.transform(validation_data[1])
        print('y shape:', Y.shape, " type: ", type(y))
        print(self.model.summary())
        if validation_data is not None:
            self.history = self.model.fit(X, Y,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                validation_split=0,
                                validation_data=(vX, vY))
        else:
            self.history = self.model.fit(X, Y,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=1,
                                     validation_split=0.2)

    def predict(self, x):
        print('Vectorizing sequence data...')
        X = self.cvect.transform(x).todense()
        print('x shape: ', X.shape, " type: ", type(x))
        return self.lbl_bin.inverse_transform(self.model.predict(X))

    def _assemble_model(self):
        print('Building model for %d classes and %d inputs ... with layers: %s' %
              (self.num_classes, self.max_words, self.layer_def)
              )
        model = Sequential()
        #input layer
        model.add(Dense(self.first_layer_size, input_shape=(self.max_words,), activation='tanh'))
        for ldef in self.layer_def:
            layer_size = ldef[0]
            num_layers = ldef[1]
            put_drops = ldef[2]
            for i in range(num_layers):
                model.add(Dense(layer_size, activation='tanh', kernel_initializer='RandomUniform', bias_initializer='random_uniform'))
                if put_drops:
                    model.add(Dropout(0.5))
        #output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model

    def _evaluate_metrics_pair(self, x: 'np.ndarray', y: 'np.ndarray', label=""):
        print("")
        print("x%s shape: " % label, x.shape, "\ty%s shape: " % label, y.shape)
        score = self.model.evaluate(x=x, y=y, batch_size=self.batch_size, verbose=1)
        print("")
        print("Score%s: " % label, score[0], "\tAccuracy%s" % label, score[1])
        preds = self.model.predict(x=x, batch_size=self.batch_size, verbose=1)
        # preds = self.lbl_bin.inverse_transform(preds)
        print(preds)
        print(y)
        print(classification_report(y_true=self.lbl_bin.inverse_transform(y), y_pred=self.lbl_bin.inverse_transform(preds)))

    def evaluation_metrics(self, x_test, y_test, x_valid, y_valid):
        xtv = self.cvect.transform(x_test).todense()
        xvv = self.cvect.transform(x_valid).todense()
        ytv = self.lbl_bin.transform(y_test)
        yvv = self.lbl_bin.transform(y_valid)
        print("validation set:")
        self._evaluate_metrics_pair(x=xvv, y=yvv, label="_valid")
        print("test set:")
        self._evaluate_metrics_pair(x=xtv, y=ytv, label="_test")

    def save_as_pickle(self, filepath: str, include_optimizer: bool=True):
        fh = open(filepath, 'wb')
        #temporarily detatch resources that we do not want saved
        model = self.model
        codes = self.codes
        titles = self.titles
        history = self.history
        self.codes = None
        self.titles = None
        self.model = None
        self.history = None
        pickle.dump(self, fh)
        #reattach resources
        self.codes = codes
        self.titles = titles
        self.model = model
        self.history = history
        if self.model is not None:
            self.model.save(filepath=filepath + '.mdl', overwrite=True, include_optimizer=include_optimizer)
        return

    @staticmethod
    def load_from_pickle(filepath: str):
        fh = open(filepath, 'rb')
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            return None
        kc = pickle.load(fh)  # type: ANNclassifier
        if os.path.exists(filepath + '.mdl') and not os.path.isdir(filepath + '.mdl'):
            kc.model = load_model(filepath + '.mdl')
        kc._load_assets()
        return kc


@click.group()
def keras_classifier_main():
    pass


@keras_classifier_main.command(name='train')
@click.argument('target_level', type=click.INT)
@click.option('--epoch', type=click.INT, default=10, help="# of epochs to use when training")
@click.option('--first_layer_size', type=click.INT, default=512, help="Size of Input Layer")
@click.option('--layer', type=(click.INT, click.INT, click.BOOL), default=(512, 1, False), multiple=True, help="triplet of values, # of neurons in layer, and # of layers, 3rd value is a bool, for whether to put a dropout layer after")
@click.option('--max_features', type=click.INT, default=10000, help="max features from count vectorizer")
@click.option('--batch_size', type=click.INT, default=64, help="batch size for tensorflow")
def keras_classifier_train(target_level, epoch, layer, max_features, first_layer_size, batch_size):
    train = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/train.set.lvl%d.P' % target_level,
                                      is_path=True).copy_and_append_empty_string_class()
    test = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/test.set.lvl%d.P' % target_level,
                                     is_path=True).copy_and_append_empty_string_class()
    # valid = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/valid.set.lvl%d.P' % target_level,
    #                                   is_path=True).copy_and_append_empty_string_class()
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
