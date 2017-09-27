import os
import pickle

from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from scribe_classifier.data.canada import AllCodes, CodeRecord, TitleSet
from typing import Tuple, List, Dict
import numpy as np


class ANNclassifier:
    def __init__(self,
                 epochs=10,
                 target_level: int=1,
                 max_words: int=20000,
                 batch_size: int=64,
                 layer_def=((512, 1, 0.0),),
                 first_layer_size=512,
                 activation='relu'):
        self.epochs = epochs
        self.activation = activation
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
        self.warmstart = False

    def set_warm_start(self, state=None):
        if state is None:
            self.warmstart = not self.warmstart
        else:
            self.warmstart = state

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
        if self.model is None or not self.warmstart:
            self._assemble_model()

    def fit(self, x, y, validation_data: 'Tuple[List[str], List[str]]'=None):
        self._setup_for_fit()
        print(self.model.summary())
        # train_batch_split = 10000
        # batches = int(len(x) / train_batch_split)
        # current_slice = 0
        # for j in range(self.epochs):
        #     for i in range(batches):
        #         if i * train_batch_split >= len(x):
        #             break
        #         slice_end = (i+1) * train_batch_split - 1
        #         current_slice = i * train_batch_split
        #         print("Slicing from %d to %d" % (current_slice, slice_end))
        #         x_slice = x[current_slice:slice_end]
        #         y_slice = y[current_slice:slice_end]
                # print('Vectorizing sequence data...')
        X = self.cvect.transform(x).todense()
        vX = self.cvect.transform(validation_data[0]).todense()
        # print('x shape: ', X.shape, " type: ", type(x))
        Y = self.lbl_bin.transform(y)
        vY = self.lbl_bin.transform(validation_data[1])
        # print('y shape:', Y.shape, " type: ", type(y))
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
        model.add(Dense(self.first_layer_size, input_shape=(self.max_words,), activation='relu'))
        for ldef in self.layer_def:
            layer_size = ldef[0]
            num_layers = ldef[1]
            put_drops = ldef[2]
            for i in range(num_layers):
                model.add(Dense(layer_size, activation=self.activation))  # kernel_initializer='RandomUniform', bias_initializer='random_uniform',))
                if put_drops:
                    model.add(Dropout(put_drops))
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
