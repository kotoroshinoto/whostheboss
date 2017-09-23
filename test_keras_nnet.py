from keras.models import load_model
from scribe_classifier.data.canada import TitleSet, TitleRecord
from scribe_classifier.data.canada import AllCodes, CodeRecord
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
from sklearn.preprocessing import LabelBinarizer


max_words = 20000
batch_size=64
all_codes = AllCodes.load_from_pickle('source_data/pickles/canada/tidy_sets/all_codes.P', is_path=True)
all_titles = TitleSet.load_from_pickle('source_data/pickles/canada/tidy_sets/all_titles.P', is_path=True)
all_texts = all_titles.get_title_vec()
target_level = 1


def classify_vect(strvect: 'List[str]', target_level=1, parent_path: str="."):
    all_title_codes = all_titles.get_code_vec(target_level=target_level)
    ac_vec = all_codes.get_codes_for_level(target_level=target_level)
    ac_vec.append('NA')
    lbl_bin = LabelBinarizer()
    lbl_bin.fit(ac_vec)
    cvect = CountVectorizer(ngram_range=(1, 6), stop_words='english', max_features=max_words)
    cvect.fit(all_texts, y=all_title_codes)
    model = load_model(filepath=parent_path+'/neural_net_level%d.frozen.mdl' % target_level)
    counts = cvect.transform(strvect).todense()
    return lbl_bin.inverse_transform(model.predict(counts, batch_size=64, verbose=1))


def class_report_nnet(target_level: int=1, parent_path: str="."):
    test = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/test.set.lvl%d.P' % target_level,
                                     is_path=True).copy_and_append_empty_string_class()
    valid = TitleSet.load_from_pickle('source_data/pickles/canada/test_sets/valid.set.lvl%d.P' % target_level,
                                      is_path=True).copy_and_append_empty_string_class()
    test_y = test.get_code_vec(target_level=target_level)
    valid_y = test.get_code_vec(target_level=target_level)
    test_pred = classify_vect(test.get_title_vec(), target_level=target_level, parent_path=parent_path)
    valid_pred = classify_vect(valid.get_title_vec(), target_level=target_level, parent_path=parent_path)
    print(classification_report(valid_y, valid_pred))
    print(classification_report(test_y, test_pred))


class_report_nnet(1, parent_path='nnmodels/')
class_report_nnet(2, parent_path='nnmodels/')
class_report_nnet(3, parent_path='nnmodels/')
