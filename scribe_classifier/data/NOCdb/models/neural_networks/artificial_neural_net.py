import os
import pickle
from typing import Tuple, List
import numpy as np
import tensorflow as tf
# from keras import regularizers
from keras.callbacks import History
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from scribe_classifier.data.NOCdb.readers import CodeSet, TitleSet


class ANNclassifier:
    """This object uses keras framework to create a neural network with specified designs,
    to learn how to classify titles"""
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
        self.history = None  # type: History
        self.codes = None
        self.titles = None
        self.lbl_bin = None  # type: LabelBinarizer
        self.cvect = None  # type: CountVectorizer
        self.num_classes = None  # type: int
        self.layer_def = layer_def
        self.first_layer_size = first_layer_size
        self.warmstart = False
        self.graph = None

    def set_warm_start(self, state=None):
        """in warm-start state the object will continue to learn, this only works if not using a frozen-save,
        as it requires the optimizer data to be able to learn"""
        if state is None:
            self.warmstart = not self.warmstart
        else:
            self.warmstart = state

    def _load_assets(self):
        """loads the all_codes and all_titles files for use by the neural network"""
        self.codes = CodeSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
        self.codes.add_emptyset()
        self.titles = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P', is_path=True).copy_and_append_empty_string_class()

    def _setup_for_fit(self):
        """sets up the classifier to be fit on data"""
        self._load_assets()
        ac_vec = self.codes.get_codes_for_level(target_level=self.target_level)
        all_texts = self.titles.get_title_vec()
        all_title_codes = self.titles.get_code_vec(target_level=self.target_level)

        if self.num_classes is None or not self.warmstart:
            self.num_classes = len(ac_vec)

        if self.lbl_bin is None or not self.warmstart:
            self.lbl_bin = LabelBinarizer()
            self.lbl_bin.fit(ac_vec)

        if self.cvect is None or not self.warmstart:
            self.cvect = CountVectorizer(ngram_range=(1, 6), stop_words='english', max_features=self.max_words)
            self.cvect.fit(all_texts, y=all_title_codes)

        if self.model is None or not self.warmstart:
            self._assemble_model()

    def fit(self, x, y, validation_data: 'Tuple[List[str], List[str]]'=None):
        """fit neural network on data, using validation data if provided"""
        self._setup_for_fit()
        print(self.model.summary())
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

    def predict(self, x, batch_size=32):
        """predict classes for input vector"""
        return self.lbl_bin.inverse_transform(self.predict_proba(x=x, batch_size=batch_size))

    def predict_proba(self, x, batch_size=32):
        """predict class probabilities for input vector"""
        # print('Vectorizing sequence data...')
        X = self.cvect.transform(x).todense()
        # print('x shape: ', X.shape, " type: ", type(x))
        if self.graph is None:
            raise RuntimeError("graph object is None")
        with self.graph.as_default():
            return self.model.predict(x=X, batch_size=batch_size)

    def _assemble_model(self):
        """Assembles neural network based on definition provided when instance was initiated"""
        # print('Building model for %d classes and %d inputs ... with layers: %s' %
        #       (self.num_classes, self.max_words, self.layer_def)
        #       )
        model = Sequential()
        #input layer
        model.add(Dense(self.first_layer_size,
                        input_shape=(self.max_words,),
                        activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
                        # activity_regularizer=regularizers.l1_l2(0.01, 0.01),
                        # bias_regularizer=regularizers.l1_l2(0.01, 0.01)
                        ))
        for ldef in self.layer_def:
            layer_size = ldef[0]
            num_layers = ldef[1]
            put_drops = ldef[2]
            for i in range(num_layers):
                model.add(Dense(layer_size,
                                activation=self.activation,
                                # kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
                                # activity_regularizer=regularizers.l1_l2(0.01, 0.01),
                                # bias_regularizer=regularizers.l1_l2(0.01, 0.01)
                                ))
                if put_drops:
                    model.add(Dropout(put_drops))
        #output layer
        model.add(Dense(self.num_classes, activation='softmax',
                        # kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
                        # activity_regularizer=regularizers.l1_l2(0.01, 0.01),
                        # bias_regularizer=regularizers.l1_l2(0.01, 0.01)
                        ))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.graph = tf.get_default_graph()

    def _evaluate_metrics_pair(self, x: 'np.ndarray', y: 'np.ndarray', label=""):
        """Evaluates some metrics for input vectors with a label"""
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
        """evaluate on provided test set and validation set"""
        valid_pred = self.predict(x_valid)
        test_pred = self.predict(x_test)
        xtv = self.cvect.transform(x_test).todense()
        xvv = self.cvect.transform(x_valid).todense()
        ytv = self.lbl_bin.transform(y_test)
        yvv = self.lbl_bin.transform(y_valid)

        print("Validation Set Report", classification_report(y_valid, valid_pred))
        print("Test Set Report", classification_report(y_test, test_pred))
        print("Val  Acc: ", accuracy_score(y_valid, valid_pred),
              "Test Acc", accuracy_score(y_test, test_pred))

        print("validation set:")
        self._evaluate_metrics_pair(x=xvv, y=yvv, label="_valid")
        print("test set:")
        self._evaluate_metrics_pair(x=xtv, y=ytv, label="_test")

    def save_as_pickle(self, filepath: str, include_optimizer: bool=True):
        """serialize object to disk, will save pickle file (*.P) for python object, and a *.P.mdl file for the keras model.
         Excluding the optimizer will result in a smaller file, but the model will not be able to learn when reloaded"""
        fh = open(filepath, 'wb')
        #temporarily detatch resources that we do not want saved
        model = self.model
        codes = self.codes
        titles = self.titles
        history = self.history
        graph = self.graph
        self.codes = None
        self.titles = None
        self.model = None
        self.history = None
        self.graph = None
        pickle.dump(self, fh)
        #reattach resources
        self.codes = codes
        self.titles = titles
        self.model = model
        self.history = history  # type: History
        self.graph = graph
        if self.model is not None:
            self.model.save(filepath=filepath + '.mdl', overwrite=True, include_optimizer=include_optimizer)
        return

    @staticmethod
    def load_from_pickle(filepath: str):
        """reconstitutes a serialized object given the path"""
        fh = open(filepath, 'rb')
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            return None
        ann_clf = pickle.load(fh)  # type: ANNclassifier
        if os.path.exists(filepath + '.mdl') and not os.path.isdir(filepath + '.mdl'):
            ann_clf.model = load_model(filepath + '.mdl')
            ann_clf.graph = tf.get_default_graph()
            if ann_clf.graph is None:
                print("default graph was None")
        ann_clf._load_assets()
        return ann_clf
