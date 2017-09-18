import pickle
import os
import sys
from csv import reader
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from canada_data.readers.codes import AllCodes
import pandas as pd


class TitleRecord:
    def __init__(self, code: str, title: str):
        self.code = code  # type: str
        self.title = title  # type: str

    def get_code(self, target_level: int):
        return self.code[0:target_level]

    def generate_compatibility_codes(self, target_level: int) -> 'List[str]':
        comp_codes = list()
        for i in range(1, 4):
            comp_codes.append(self.get_code(target_level=target_level))
        return comp_codes

    def combine_text(self, all_codes: 'AllCodes', target_level: int) -> 'str':
        if len(self.title) == 0:
            return ""
        comp_codes = self.generate_compatibility_codes(target_level=target_level)
        text = []
        for i in range(target_level, 4):
            coderecord = all_codes.codes[comp_codes[i-1]]
            text.append(coderecord.desc)
        text.append(self.title)
        return " ".join(text)


class TitleSet:
    def __init__(self):
        self.records = list()  # type: List[TitleRecord]

    def to_dataframe(self, target_level=1) -> 'pd.DataFrame':
        titles, codes = self.split_into_title_and_code_vecs(target_level=target_level)
        title_series = pd.Series(titles)
        code_series = pd.Series(codes)
        df = pd.DataFrame()
        df['codes'] = code_series
        df['titles'] = title_series
        return df

    def count_classes(self, target_level=1)->'Dict[str, int]':
        counts = dict()
        for record in self.records:
            code = record.get_code(target_level)
            if code not in counts:
                counts[code] = 1
            else:
                counts[code] += 1
        return counts

    def add_titles_from_file(self, filename):
        titles = reader(open(filename, 'r'), dialect='excel-tab')
        for title_record in titles:
            self.records.append(TitleRecord(title_record[0], title_record[1]))

    def add_titles_from_vecs(self, title_vec, code_vec):
        numtitle = len(title_vec)
        numcodes = len(code_vec)
        if numtitle != numcodes:
            raise ValueError("title vector and code vector must be same size")
        if numtitle == 0:
            return
        for i in range(numtitle):
            self.records.append(TitleRecord(title=title_vec[i], code=code_vec[i]))

    def get_title_vec(self) -> 'List[str]':
        vec = list()
        for record in self.records:
            vec.append(record.title)
        return vec

    def get_code_vec(self, target_level=4 ) -> 'List[str]':
        vec = list()
        for record in self.records:
            vec.append(record.get_code(target_level))
        return vec

    def split_into_title_and_code_vecs(self, target_level=4):
        title_vec = list()
        code_vec = list()
        for record in self.records:
            title_vec.append(record.title)
            code_vec.append(record.get_code(target_level))
        return title_vec, code_vec

    @classmethod
    def from_vecs_split_data_valid_train_test(cls, title_vec, code_vec, test_split=0.20, valid_split=0.20) -> 'Tuple[TitleSet, TitleSet, TitleSet]':
        new_set = cls()
        new_set.add_titles_from_vecs(title_vec=title_vec, code_vec=code_vec)
        return new_set.split_data_valid_train_test(test_split=test_split, valid_split=valid_split)

    def split_data_valid_train_test(self, target_level=1, test_split=0.20, valid_split=0.20) -> 'Tuple[TitleSet, TitleSet, TitleSet]':
        # split datasets into train/validation/test
        title_vec, code_vec = self.split_into_title_and_code_vecs()
        target_codes = self.get_code_vec(target_level=target_level)
        # print("examples: %d\tcodes: %d" % (len(title_vec), len(code_vec)))
        split_prop = (test_split + valid_split)
        split2_prop = test_split / split_prop
        # print("initial split: %.2f, second split: %.2f" % (split_prop, split2_prop))
        title_train, title_split, code_train, code_split = train_test_split(
            title_vec,
            code_vec,
            stratify=target_codes,
            test_size=split_prop
        )
        # save this for return values
        train = self.__class__()
        train.add_titles_from_vecs(title_vec=title_train, code_vec=code_train)
        #create this to generate stratify list
        split = self.__class__()
        split.add_titles_from_vecs(title_vec=title_split, code_vec=code_split)
        split_target_codes = split.get_code_vec(target_level=target_level)
        title_valid, title_test, code_valid, code_test = train_test_split(
            title_split,
            code_split,
            stratify=split_target_codes,
            test_size=split2_prop
        )
        # put validation set back into training set
        title_train += title_valid
        code_train += code_valid

        # save these for return values
        valid = self.__class__()
        valid.add_titles_from_vecs(title_vec=title_valid, code_vec=code_valid)
        test = self.__class__()
        test.add_titles_from_vecs(title_vec=title_test, code_vec=code_test)
        return train, valid, test

    @classmethod
    def from_files(cls, example_file, target_level, code_file=None, append_empty_class=False):
        new_set = cls()
        new_set.add_titles_from_file(example_file)
        if code_file is not None:
            all_codes = AllCodes()
            all_codes.add_codes_from_file(filename=code_file)
            new_set = new_set.generate_combined(codes=all_codes)
            (xvec, yvec) = generate_combined(level_codes, title_records, target_level)
        else:
            (xvec, yvec) = generate_uncombined_text_for_target_level(title_records, target_level)
        if append_empty_class:
            newobj.append_empty_string_class()
        return newobj

    def save_as_pickle(self, file, is_path=False):
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'TitleSet':
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        ds = pickle.load(handle)
        handle.close()
        return ds

    def append_empty_string_class(self, label='NA'):
        num_to_add = int(len(self.records)/4)
        for i in range(num_to_add):
            self.records.append(TitleRecord(title="", code=label))

    def generate_combined(self, codes: 'AllCodes', target_level: int) -> 'TitleSet':
        new_set = TitleSet()
        for trecord in self.records:
            new_trecord = TitleRecord(code=trecord.code, title=trecord.combine_text(all_codes=codes,
                                                                                    target_level=target_level))
            new_set.records.append(new_trecord)
        return new_set
