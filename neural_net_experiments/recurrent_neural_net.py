import pickle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from scribe_classifier.data.canada.NOCdb import AllCodes, CodeRecord, TitleSet, TitleRecord
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

    def fit(self):
        #Convert text samples in dataset to sequences of word indexes (encoding/bagofwords, etc)
        all_codes = AllCodes.load_from_pickle('source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
        all_codes.add_emptyset("NA")
        tset = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P', is_path=True)
        all_codevec = all_codes.get_codes_for_level(target_level=self.target_level)
        num_classes = len(all_codevec)
        tset = tset.copy_and_append_empty_string_class()
        tset_codevec = tset.get_code_vec(target_level=self.target_level)
        tset_titlevec = tset.get_title_vec()
        print("%d titles" % len(tset_titlevec))
        print("%d codes" % len(tset_codevec))

        self.lblbin.fit(all_codevec)
        self.tokenizer.fit_on_texts(tset_titlevec)
        labels = self.lblbin.transform(tset_codevec)
        print('%s texts' % len(tset_titlevec))
        sequences = self.tokenizer.texts_to_sequences(tset_titlevec)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences=sequences, maxlen=self.max_sequence_length)
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        x_train, x_val, y_train, y_val = train_test_split(
            data,
            labels,
            stratify=tset_codevec,
            test_size=0.1,
            shuffle=True
        )

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
        self.assemble_model(embedding_layer=embedding_layer, num_classes=num_classes)
        self.model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=self.epochs, batch_size=self.batch_size)
        train_scores = self.model.evaluate(x_train, y_train)
        scores = self.model.evaluate(x_val, y_val)
        print("Training Accuracy: %.2f%%" % (train_scores[1] * 100))
        print("Test     Accuracy: %.2f%%" % (scores[1] * 100))

    def predict(self, X):
        pass

    def save_as_pickle(self, filepath: str, include_optimizer: bool=True):
        fh = open(filepath, 'wb')
        #temporarily detatch resources that we do not want saved
        model = self.model
        history = self.history
        self.model = None
        self.history = None
        pickle.dump(self, fh)
        #reattach resources
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
        kc = pickle.load(fh)
        if os.path.exists(filepath + '.mdl') and not os.path.isdir(filepath + '.mdl'):
            kc.model = load_model(filepath + '.mdl')
        kc._load_assets()
        return kc


def recurrent_neural_net_main():
    target_level = 3

    rnc = RecurrentNeuralClassifier(embeddings_file='source_data/wiki_glove/glove.6B.100d.txt',
                                    embedding_dim=100,
                                    target_level=target_level,
                                    word_limit=20000,
                                    max_len=1000,
                                    epochs=15,
                                    batch_size=256
                                    )

    rnc.fit()
    rnc.save_as_pickle("nnmodels/rnnclf_lvl%d.P" % target_level)


if __name__ == "__main__":
    recurrent_neural_net_main()
