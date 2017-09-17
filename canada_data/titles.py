import pickle
from _csv import reader
from sklearn.model_selection import train_test_split
from canada_data.codes import AllCodes


class TitleRecord:
    def __init__(self, code: str, title: str):
        self.code = code.lower()  # type: str
        self.title = title.lower()  # type: str

    def get_code(self, level: int):
        return self.code[0:level]

    def combine_text(self, code_dict:'Dict[str, str]', target_level:int) -> 'str':
        combined = dict()
        # target_code = self.get_code(target_level)
        # print(target_code)
        text = []
        for i in range(target_level, 4):
            text.append(code_dict[self.get_code(i)])
        text.append(self.title)
        return " ".join(text)


class TitleSet:
    def __init__(self, titles, codes):
        self.titles = titles
        self.codes = codes

    def count_classes(self)->'Dict[str, int]':
        counts = dict()
        for cat in self.codes:
            if cat not in counts:
                counts[cat] = 1
            else:
                counts[cat] += 1
        return counts

    def _read_titles_from_file(self, filename) -> 'List[TitleRecord]':
        titles = reader(open(filename), dialect='excel-tab')
        records = []
        for title_record in titles:
            records.append(TitleRecord(title_record[0], title_record[1]))
            return records

    @classmethod
    def from_vecs_split_data_valid_train_test(cls, title_vec, code_vec, test_split=0.20, valid_split=0.20) -> 'Tuple[TitleSet, TitleSet, TitleSet]':
        return cls(title_vec, code_vec).split_data_valid_train_test(test_split=test_split, valid_split=valid_split)

    def split_data_valid_train_test(self, test_split=0.20, valid_split=0.20) -> 'Tuple[TitleSet, TitleSet, TitleSet]':
        # split datasets into train/validation/test
        title_train, title_split, code_train, code_split = train_test_split(self.titles, self.codes,
                                                              stratify=self.codes,
                                                              test_size=(test_split + valid_split))

        title_valid, title_test, code_valid, code_test = train_test_split(title_split, code_split,
                                                            stratify=code_split,
                                                            test_size=(test_split / (test_split + valid_split)))
        # put validation set back into training set
        title_train += title_valid
        code_train += code_valid

        # save these for return values
        train = self.__class__(titles=title_train, codes=code_train)
        valid = self.__class__(titles=title_valid, codes=code_valid)
        test = self.__class__(titles=title_test, codes=code_test)
        return train, valid, test

    @classmethod
    def from_files(cls, example_file, target_level, code_file=None, append_empty_class=False):
        title_records = read_titles(example_file, target_level)
        if code_file is not None:
            codes = AllCodes.from_file(code_file)
            (xvec, yvec) = generate_combined(level_codes, title_records, target_level)
        else:
            (xvec, yvec) = generate_uncombined_text_for_target_level(title_records, target_level)
        newobj = cls(xvec, yvec)
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
        num_to_add = int(len(self.titles)/4)
        for i in range(num_to_add):
            self.titles.append("")
            self.codes.append(label)

    def generate_combined(self, codes: 'AllCodes') -> 'TitleSet':
        pass


def generate_combined(level_codes: 'Dict[str, str]', title_records: 'List[TitleRecord]', target_level) -> 'Tuple[List[str], List[str]]':
    x_list = []
    y_list = []
    for trecord in title_records:
        y_list.append(trecord.get_code(target_level))
        x_list.append(trecord.combine_text(level_codes, target_level))
    return x_list, y_list


def generate_uncombined_text_for_target_level(title_records: 'List[TitleRecord]', target_level) -> 'Tuple[List[str], List[str]]':
    x_list = []
    y_list = []
    for trecord in title_records:
        y_list.append(trecord.get_code(target_level))
        x_list.append(trecord.title)
    return x_list, y_list