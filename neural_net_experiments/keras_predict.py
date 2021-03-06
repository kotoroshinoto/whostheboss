'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

# from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from scribe_classifier.data.scribe.readers.dbhandler import DataFramePickler
from scribe_classifier.data.NOCdb import TitleSet, TitleRecord
from scribe_classifier.data.NOCdb import AllCodes, CodeRecord
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.models import load_model


BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/source_data/wiki_glove/'
TEXT_DATA_DIR = BASE_DIR + '/source_data/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')


texts = []  # list of text samples
labels = []  # list of label ids
labels_index = {}  # dictionary mapping label name to numeric id

target_level = 3

all_codes = AllCodes.load_from_pickle('source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
ac_vec = all_codes.get_codes_for_level(target_level=target_level)
lbl_enc = LabelEncoder()
tset = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P',is_path=True)
tset = tset.copy_and_append_empty_string_class()
ac_vec.append('NA')
texts = tset.get_title_vec()
text_labels = tset.get_code_vec(target_level=target_level)
labels = lbl_enc.fit_transform(text_labels)


for i in range(len(ac_vec)):
    labels_index[i] = ac_vec[i]

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
split_out = train_test_split(data, labels, train_size=0.20, shuffle=True, stratify=text_labels)
x_val = split_out[0]
x_train = split_out[1]
y_val = split_out[2]
y_train = split_out[3]
print("train_X: ", x_train.shape)
print("train_Y: ", y_train.shape)
print("val_X", x_val.shape)
print("val_Y", y_val.shape)

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))

model = load_model("level%dconvnn.P" % target_level, compile=True)

train_pred = model.predict(x=x_train, batch_size=128)
val_pred = model.predict(x=x_val, batch_size=128)