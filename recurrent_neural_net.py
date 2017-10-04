import pickle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from scribe_classifier.data.NOCdb.readers import CodeSet, CodeRecord, TitleSet, TitleRecord
from typing import List, Dict, Tuple
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import os
import os.path
import numpy as np
import click


class RecurrentNeuralClassifier:
    def __init__(self, embeddings_file: 'str', embedding_dim: int, word_limit=20000, max_len=1000, target_level=1, epochs=5, batch_size=64):
        self.target_level = target_level
        self.word_limit = word_limit
        self.embedding = embeddings_file
        self.lblbin = LabelBinarizer()
        self.tokenizer = Tokenizer(num_words=self.word_limit)
        self.max_sequence_length = max_len
        self.embedding_dim = embedding_dim
        self.model = None  # type: Sequential
        self.history = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.warm_start = False

    def set_warmstart(self, value=True):
        self.warm_start = value

    @staticmethod
    def load_embeddings(embeddings_file):
        embeddings_index = dict()
        f = open(embeddings_file, 'r')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def build_neural_net(self):
        pass

    def assemble_model(self, embedding_layer: 'Embedding', num_classes: int):
        model = Sequential()
        model.add(embedding_layer)
        # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(150, dropout=1.0/3.0, recurrent_dropout=1.0/3.0))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def fit(self, X, y, validation_data:'Tuple[List[str], List[str]]'=None):
        #Convert text samples in dataset to sequences of word indexes (encoding/bagofwords, etc)
        all_codes = CodeSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
        all_codes.add_emptyset("NA")
        tset = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P', is_path=True)
        all_codevec = all_codes.get_codes_for_level(target_level=self.target_level)
        num_classes = len(all_codevec)
        tset = tset.copy_and_append_empty_string_class()
        tset_titlevec = tset.get_title_vec()

        #fit label binarizer and tokenizer on ALL the data
        self.lblbin.fit(all_codevec)
        self.tokenizer.fit_on_texts(tset_titlevec)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        #transform our actual data
        y_train = self.lblbin.transform(y)

        x_train = pad_sequences(sequences=self.tokenizer.texts_to_sequences(X),
                             maxlen=self.max_sequence_length)

        print('%s texts' % len(X))
        print('Shape of data tensor:', x_train.shape)
        print('Shape of label tensor:', y_train.shape)

        embeddings_index = self.load_embeddings(self.embedding)
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        embedding_layer = Embedding(len(word_index) + 1,
                                    self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        if not self.warm_start:
            self.assemble_model(embedding_layer=embedding_layer, num_classes=num_classes)
        if validation_data is not None:
            y_val = self.lblbin.transform(validation_data[1])
            x_val = pad_sequences(sequences=self.tokenizer.texts_to_sequences(validation_data[0]),
                             maxlen=self.max_sequence_length)
            self.model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=self.epochs, batch_size=self.batch_size)
            train_scores = self.model.evaluate(x_train, y_train)
            print("Training Accuracy: %.4f%%" % (train_scores[1] * 100))
            scores = self.model.evaluate(x_val, y_val)
            print("Test     Accuracy: %.4f%%" % (scores[1] * 100))
        else:
            self.model.fit(x=x_train, y=y_train, epochs=self.epochs, batch_size=self.batch_size)
            train_scores = self.model.evaluate(x_train, y_train)
            print("Training Accuracy: %.4f%%" % (train_scores[1] * 100))

    def predict(self, X):
        pass

    def save_as_pickle(self, filepath: str, include_optimizer: bool=True):
        fh = open(filepath, 'wb')
        #temporarily detatch resources that we do not want saved
        model = self.model
        history = self.history
        ws = self.warm_start
        self.model = None
        self.history = None
        self.warm_start = False
        pickle.dump(self, fh)
        #reattach resources
        self.model = model
        self.history = history
        self.warm_start = ws
        if self.model is not None:
            self.model.save(filepath=filepath + '.mdl', overwrite=True, include_optimizer=include_optimizer)
        return

    @staticmethod
    def load_from_pickle(filepath: str) -> 'RecurrentNeuralClassifier':
        fh = open(filepath, 'rb')
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            return None
        kc = pickle.load(fh)
        if os.path.exists(filepath + '.mdl') and not os.path.isdir(filepath + '.mdl'):
            kc.model = load_model(filepath + '.mdl')
        # kc._load_assets()
        return kc


@click.command()
@click.option('--epochs', type=click.INT, default=30, help="# of keras model epochs to rain")
@click.option('--emptyset', type=click.STRING, default=None, help="Add Empty String Dataset with given label to test set and validation set  before making predictions, if you provide an empty string label, default 'NA' will be used instead")
@click.option('--batch_size', type=click.INT, default=32, help="keras batch size")
@click.option('--word_limit', type=click.INT, default=20000, help="word limit for tokenizer")
@click.option('--max_len', type=click.INT, default=1000, help="maximum length for embedding layer")
@click.option('--embed_dim', type=click.INT, default=300, help="embedding dim from wikipedia 50, 100, 200, 300")
@click.option('--target_level', type=click.IntRange(1, 4), default=1, help="NOC code level to make model predict")
@click.option('--model_filepath', type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True), help="path to use to save model")
@click.option('--train_filepath', type=click.File('rb'), required=True, help="point to pickle containing training data")
@click.option('--test_filepath', type=click.File('rb'), required=True, help="point to pickle containing test data for validation")
@click.option('--warmstart/--no-warmstart', default=False, help='continue training existing model for more epochs')
def recurrent_neural_net_main(epochs, batch_size, word_limit, max_len, embed_dim, target_level, model_filepath, train_filepath, test_filepath, emptyset, warmstart):
    if emptyset == "":
        emptyset = "NA"

    train_set = TitleSet.load_from_pickle(file=train_filepath, is_path=False).copy_and_append_empty_string_class(label=emptyset)
    test_set = TitleSet.load_from_pickle(file=test_filepath, is_path=False).copy_and_append_empty_string_class(label=emptyset)

    train_tvec = train_set.get_title_vec()
    train_codes = train_set.get_code_vec(target_level=target_level)
    test_tvec = test_set.get_title_vec()
    test_codes = test_set.get_code_vec(target_level=target_level)
    if warmstart:
        print("loading existing model for more training")
        rnc = RecurrentNeuralClassifier.load_from_pickle(filepath=model_filepath)
        rnc.set_warmstart()
    else:
        print("assembling new model")
        rnc = RecurrentNeuralClassifier(embeddings_file='source_data/wiki_glove/glove.6B.300d.txt',
                                        embedding_dim=embed_dim,
                                        target_level=target_level,
                                        word_limit=word_limit,
                                        max_len=max_len,
                                        epochs=epochs,
                                        batch_size=batch_size
                                        )
    rnc.fit(X=train_tvec, y=train_codes, validation_data=(test_tvec, test_codes))
    rnc.save_as_pickle(model_filepath)


if __name__ == "__main__":
    recurrent_neural_net_main()
