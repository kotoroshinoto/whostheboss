# from  tensorflow.contrib.learn import DNNClassifier, SKCompat
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from scribe_classifier.data.canada import AllCodes, TitleSet
from scribe_classifier.util import ObjectPickler

all_codes = AllCodes.load_from_pickle("./source_data/pickles/canada/tidy_sets/all_codes.P", is_path=True)

target_level = 3
emptyset_label = "NA"

all_code_vec = sorted(all_codes.get_codes_for_level(target_level=target_level))
if emptyset_label is not None:
    all_code_vec.append(emptyset_label)

print(all_code_vec)
lblenc = LabelEncoder()
lblenc.fit(all_code_vec)

print(lblenc.transform(all_code_vec))

data = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P', is_path=True)
data = data.copy_and_append_empty_string_class(label=emptyset_label, prop_records=0.25)
data_title_vec = data.get_title_vec()
data_code_vec = data.get_code_vec(target_level=target_level)

kfold = KFold(n_splits=10, shuffle=True)
vectr = CountVectorizer(stop_words='english', ngram_range=(1, 5))
X = vectr.fit_transform(data_title_vec).todense()
feature_names = vectr.get_feature_names()
Y = lblenc.transform(data_code_vec)
catY = np_utils.to_categorical(Y)

features = X.shape[1]
print(features)
catcount = len(all_code_vec)
print(catcount)
num_hidden = catcount*50

#
# def baseline_model():
#     model = Sequential()
#     model.add(Dense(catcount*2, input_dim=features, activation='relu'))
#     model.add(Dense(catcount*2, activation='relu'))
#     model.add(Dense(catcount, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# max_input_length = -1

embedding_vector_length = 32


def recurrent_model():
    model = Sequential()
    model.add(Embedding(features, embedding_vector_length, input_length=features))
    model.add(LSTM(catcount*100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(catcount, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
estimator = KerasClassifier(build_fn=recurrent_model, epochs=4, batch_size=64, verbose=1)
# results = cross_val_score(estimator, X, catY, cv=stratk)
estimator.fit(x=X, y=catY)
ObjectPickler.save_as_pickle(estimator, "./nn_object.P")

