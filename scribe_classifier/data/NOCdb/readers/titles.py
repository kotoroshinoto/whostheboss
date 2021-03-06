import pickle
import re
from csv import reader
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from slugify import slugify
from .codes import CodeSet, CodeRecord
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle


class TitleRecord:
    """This class represents a single title & code pair from the NOC database"""
    def __init__(self, code: str, title: str, is_emptyset=False):
        self.code = code  # type: str
        self.title = title  # type: str
        self.is_emptyset = is_emptyset

    def get_code(self, target_level: int):
        """get the code for this record"""
        if self.is_emptyset:
            return self.code
        else:
            return self.code[0:target_level]

    def generate_compatibility_codes(self, target_level: int) -> 'List[str]':
        """generate a list of compatibility codes"""
        comp_codes = list()
        for i in range(1, 4):
            comp_codes.append(self.get_code(target_level=target_level))
        return comp_codes


class TitleSet:
    """This class represents an entire set of title records"""
    def __init__(self):
        """initializes as empty and without an emptyset label"""
        self.records = list()  # type: List[TitleRecord]
        self.emptyset_label = None

    def to_dataframe(self, target_level=1) -> 'pd.DataFrame':
        """get a dataframe containing the information from the title set"""
        titles, codes = self.split_into_title_and_code_vecs(target_level=target_level)
        title_series = pd.Series(titles)
        code_series = pd.Series(codes)
        df = pd.DataFrame()
        df['codes'] = code_series
        df['titles'] = title_series
        return df

    def count_classes(self, target_level=1)->'Dict[str, int]':
        """count the number of appearances of classes in this title set"""
        counts = dict()
        for record in self.records:
            code = record.get_code(target_level)
            if code not in counts:
                counts[code] = 1
            else:
                counts[code] += 1
        return counts

    def add_title(self, title_record):
        """add a new title record"""
        self.records.append(title_record)

    def add_titles_from_file(self, filename):
        """add all titles from target file, to be used with a cleaned version of the NOC database"""
        titles = reader(open(filename, 'r'), dialect='excel-tab')
        for title_record in titles:
            self.records.append(TitleRecord(title_record[0], title_record[1]))

    def add_titles_from_vecs(self, title_vec, code_vec):
        """adds titles from a title/code vector pair"""
        numtitle = len(title_vec)
        numcodes = len(code_vec)
        if numtitle != numcodes:
            raise ValueError("title vector and code vector must be same size")
        if numtitle == 0:
            return
        for i in range(numtitle):
            self.records.append(TitleRecord(title=title_vec[i], code=code_vec[i]))

    def get_title_vec(self) -> 'List[str]':
        """get a title vector"""
        vec = list()
        for record in self.records:
            vec.append(record.title)
        return vec

    def get_code_vec(self, target_level=4 ) -> 'List[str]':
        """get a code vector for target level"""
        vec = list()
        for record in self.records:  # type: TitleRecord
            if record.code == self.emptyset_label:
                vec.append(self.emptyset_label)
            else:
                vec.append(record.get_code(target_level))
        return vec

    def split_into_title_and_code_vecs(self, target_level=4):
        """split into title and code vectors"""
        title_vec = list()
        code_vec = list()
        for record in self.records:
            title_vec.append(record.title)
            code_vec.append(record.get_code(target_level))
        return title_vec, code_vec

    def split_data_train_test(self, target_level=1, test_split=0.20) -> 'Tuple[TitleSet, TitleSet]':
        """split records once to produce a train and test set"""
        title_vec, code_vec = self.split_into_title_and_code_vecs()
        target_codes = self.get_code_vec(target_level=target_level)
        title_train, title_test, code_train, code_test = train_test_split(
            title_vec,
            code_vec,
            stratify=target_codes,
            test_size=test_split
        )
        train = self.__class__()
        train.emptyset_label = self.emptyset_label
        train.add_titles_from_vecs(title_vec=title_train, code_vec=code_train)
        test = self.__class__()
        test.emptyset_label = self.emptyset_label
        test.add_titles_from_vecs(title_vec=title_test, code_vec=code_test)
        return train, test

    def split_data_valid_train_test(self, target_level=1, test_split=0.20, valid_split=0.20) -> 'Tuple[TitleSet, TitleSet, TitleSet]':
        """split records twice to produce a training set, a test set, and a smaller validation set"""
        # split datasets into train/validation/test
        title_vec, code_vec = self.split_into_title_and_code_vecs()
        target_codes = self.get_code_vec(target_level=target_level)
        # print("examples: %d\tcodes: %d" % (len(title_vec), len(code_vec)))
        # print("initial split: %.2f, second split: %.2f" % (split_prop, split2_prop))
        title_train, title_test, code_train, code_test = train_test_split(
            title_vec,
            code_vec,
            stratify=target_codes,
            test_size=test_split
        )
        train_prop = 1.0-test_split
        # save this for return values
        train = self.__class__()
        train.add_titles_from_vecs(title_vec=title_train, code_vec=code_train)
        train_target_codes = train.get_code_vec(target_level=target_level)
        train.emptyset_label = self.emptyset_label
        # for instance with test 0.2 and valid 0.2, train will be 0.8, so we need 0.2 / 0.8 = 0.25 to apply against 0.8
        # to get the same # of items as the original 0.2 against 1.0
        title_smaller_train, title_valid, code_smaller_train, code_valid = train_test_split(
            title_train,
            code_train,
            stratify=train_target_codes,
            test_size=valid_split/train_prop
        )

        # save these for return values
        valid = self.__class__()
        valid.emptyset_label = self.emptyset_label
        valid.add_titles_from_vecs(title_vec=title_valid, code_vec=code_valid)
        test = self.__class__()
        test.emptyset_label = self.emptyset_label
        test.add_titles_from_vecs(title_vec=title_test, code_vec=code_test)
        return train, valid, test

    def save_as_pickle(self, file, is_path=False):
        """save this title set as a pickle"""
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'TitleSet':
        """load a title set from a pickle"""
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        ds = pickle.load(handle)
        handle.close()
        return ds

    def copy_and_append_empty_string_class(self, label='NA', prop_records=0.25) -> 'TitleSet':
        """create a new copy of this title set and apply an empty string class to it, creating size * proportion
        # of empty string records with the given code label"""
        if self.emptyset_label is not None:
            raise ValueError("Already has an empty_set")
        if label is not None and label == "":
            label = "NA"
        new_copy = self.__class__()  # type: TitleSet
        new_copy.records = list(self.records)
        num_to_add = int(len(self.records) * prop_records)
        for i in range(num_to_add):
            new_copy.records.append(TitleRecord(title="", code=label, is_emptyset=True))
        new_copy.emptyset_label = label
        return new_copy

    def get_sets_for_fitting_multi_level(self, all_codes: 'CodeSet', target_level=1) -> 'Dict[str, TitleSet]':
        """
        This function will assemble the target codes for fitting title_set for classifier at various levels
         and with given parent code. Indexable by parent_code
        """
        sets_for_codes = dict()  # type: Dict[str, TitleSet]
        # empty string for no parent (first level)
        sets_for_codes[""] = TitleSet()
        for code_key in all_codes.codes:  # type: CodeRecord
            code = all_codes.codes[code_key]
            code_level = code.get_level()
            if code_level < target_level:
                # print("code_key: %s" % code_key)
                sets_for_codes[code.code] = TitleSet()

        for title_record in self.records:
        # for each record
            for previous_level in range(target_level):
                # for each level
                parent_code = title_record.get_code(previous_level)
                if len(title_record.title) == 0:
                    target_code = title_record.code
                else:
                    target_code = title_record.get_code(previous_level + 1)
                # print("%s\t%s" % (parent_code, target_code))
                sets_for_codes[parent_code].add_title(title_record)
        return sets_for_codes

    def copy_and_oversample_to_flatten_stratification(self, target_level=4) -> 'TitleSet':
        """a strategy I attempted was oversampling to get more input data and hopefully prevent the small number of some
        categories from being too much of a problem. This didn't work very well, but might deserve additional thought, so
        I left it in the codebase"""
        counts = self.count_classes(target_level=target_level)
        max_count = -1
        for code in counts:
            if counts[code] > max_count:
                max_count = counts[code]
        target_count = max_count * 10
        ratio_dict = dict()
        for code in counts:
            ratio_dict[code] = target_count - counts[code]
        title_enc = LabelEncoder()
        code_enc = LabelEncoder()
        # ros = RandomOverSampler(ratio='all')
        ros = RandomOverSampler(ratio=ratio_dict)
        code_vec = self.get_code_vec(target_level=4)
        # code_vec_encoded = code_enc.fit_transform(code_vec)
        title_vec = self.get_title_vec()
        # title_vec_encoded = title_enc.fit_transform(title_vec)

        # X = np.array(title_vec_encoded, dtype=np.int).reshape(-1, 1)
        # Y = np.array(code_vec_encoded, dtype=np.int)

        X_resamp, Y_resamp = ros.fit_sample(X=np.asarray(title_vec).reshape(-1, 1), y=code_vec)

        new_set = self.__class__()  # type: TitleSet
        new_set.emptyset_label = self.emptyset_label
        for trecord in self.records:
            new_set.add_title(title_record=TitleRecord(
                code=trecord.code,
                title=trecord.title,
                is_emptyset=trecord.is_emptyset)
            )

        # x_rs_vec = list(title_enc.inverse_transform(X_resamp.flatten()))
        # y_rs_vec = list(code_enc.inverse_transform(Y_resamp))

        # new_set.add_titles_from_vecs(title_vec=x_rs_vec, code_vec=y_rs_vec)
        new_set.add_titles_from_vecs(title_vec=X_resamp.flatten().tolist(), code_vec=Y_resamp.tolist())
        # print(new_set.count_classes(target_level=target_level))
        return new_set


class TitlePreprocessor:
    """This class performs some preprocessing of titles"""
    male_titles = [
        'man', 'master', 'actor', 'host', 'waiter', 'headwaiter', 'lord', 'boy', 'brother', 'men'
    ]
    female_titles = [
        'woman', 'mistress', 'actress', 'hostess', 'waitress', 'headwaitress', 'lady', 'girl', 'sister', 'women'
    ]
    central_gender_re_str = "(%s)[/](%s)" % ("|".join(male_titles), "|".join(female_titles))
    reverse_central_gender_re_str = "(%s)[/](%s)" % ("|".join(female_titles),"|".join(male_titles))
    gender_re = re.compile("(.*)(\S*)%s(.*)" % central_gender_re_str)
    reverse_gender_re = re.compile("(.*)(\S*)%s(.*)" % central_gender_re_str)
    double_gender_re = re.compile("(.*)(\S*)%s(.*)(\S*)%s(.*)" % (central_gender_re_str, central_gender_re_str))
    prefix_re = re.compile("(^|.*\s)(co)[-](\S+)(.*)")

    @classmethod
    def preprocess_title_split_genders(cls, t: 'str') -> 'List[str]':
        """splits a dual gender title into separate entries, properly recombining the text"""
        match_obj = cls.double_gender_re.match(t)
        if match_obj:
            mog = match_obj.groups()
            title_m = "".join([mog[0], mog[1], mog[2], mog[4], mog[5], mog[6], mog[8]])
            title_f = "".join([mog[0], mog[1], mog[3], mog[4], mog[5], mog[7], mog[8]])
            return [title_m, title_f]

        match_obj = cls.gender_re.match(t)
        if match_obj:
            mog = match_obj.groups()
            title_m = "%s%s%s%s" % (mog[0], mog[1], mog[2], mog[4])
            title_f = "%s%s%s%s" % (mog[0], mog[1], mog[3], mog[4])
            return [title_m, title_f]

        match_obj = cls.reverse_gender_re.match(t)
        if match_obj:
            mog = match_obj.groups()
            title_f = "%s%s%s%s" % (mog[0], mog[1], mog[2], mog[4])
            title_m = "%s%s%s%s" % (mog[0], mog[1], mog[3], mog[4])
            return [title_m, title_f]

        return [t]

    @classmethod
    def preprocess_titleset_split_genders(cls, tset: 'TitleSet') -> 'TitleSet':
        """splits the dual gender titles into separate entries, properly recombining the text"""
        new_tset = TitleSet()
        for trecord in tset.records:  # type: 'TitleRecord'
            split_result = cls.preprocess_title_split_genders(trecord.title)
            new_tset.add_title(TitleRecord(code=trecord.code,
                                           title=split_result[0],
                                           is_emptyset=trecord.is_emptyset))
            if len(split_result) == 2:
                new_tset.add_title(TitleRecord(code=trecord.code,
                                               title=split_result[1],
                                               is_emptyset=trecord.is_emptyset))
        return new_tset

    @classmethod
    def preprocess_title_prefixes(cls, t: 'str') -> 'str':
        """reduce prefix by removing special characters such as co-ordinator -> coordinator"""
        match_obj = cls.prefix_re.match(t)
        # squash the text by removing the dash
        if match_obj:
            mog = match_obj.groups()
            return "".join(mog)
        else:
            return t

    @classmethod
    def preprocess_titleset_prefixes(cls, tset: 'TitleSet') -> 'TitleSet':
        """reduce prefixes by removing their special characters such as co-ordinator -> coordinator"""
        new_tset = TitleSet()
        for trecord in tset.records:  # type: 'TitleRecord'
            new_title = cls.preprocess_title_prefixes(trecord.title)
            new_tset.add_title(TitleRecord(code=trecord.code,
                                           title=new_title,
                                           is_emptyset=trecord.is_emptyset))
        return new_tset

    @staticmethod
    def preprocess_slugify(s: str)->str:
        """slugify a string, using spaces as separators"""
        s = slugify(s, separator=" ")
        return s

    @classmethod
    def preprocess_slugify_titleset(cls, tset: 'TitleSet') -> 'TitleSet':
        """slugify titles in a title set, using spaces as separators"""
        new_tset = TitleSet()
        for trecord in tset.records:
            new_tset.add_title(TitleRecord(code=trecord.code,
                                           title=cls.preprocess_slugify(trecord.title),
                                           is_emptyset=trecord.is_emptyset))
        return new_tset

    @staticmethod
    def tokenize_title(t: str, remove_stopwords=True)-> 'List[str]':
        """tokenize this title"""
        toks = t.split()
        if remove_stopwords:
            new_toks = []
            for tok in toks:
                if tok.lower() not in ENGLISH_STOP_WORDS:
                    new_toks.append(tok)
            return new_toks
        else:
            return toks

    @classmethod
    def tokenize_titleset(cls, tset: 'TitleSet', remove_stopwords=True) ->'List[List[str]]':
        """tokenize entire title set"""
        tvec = tset.get_title_vec()
        toks = []
        for t in tvec:
            toks.append(cls.tokenize_title(t))
        return toks
