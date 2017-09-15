import pickle
from sklearn.model_selection import train_test_split
from canada_data.combine_strings import *
from sklearn import preprocessing


class DataSet:
    def __init__(self, X, Y, encoder=None):
        self.X = X
        self.Y = Y
        if encoder is None:
            self.encoder = preprocessing.LabelEncoder() # type: preprocessing.LabelEncoder
            self.encoder.fit(self.Y)
        else:
            self.encoder = encoder # type: preprocessing.LabelEncoder
        self.Y_enc = self.encoder.transform(self.Y)

    def count_classes(self)->'Dict[str, int]':
        counts = dict()
        for cat in self.Y:
            if cat not in counts:
                counts[cat] = 1
            else:
                counts[cat] += 1
        return counts

    @classmethod
    def from_vecs_split_data_valid_train_test(cls, xvec, yvec, test_split=0.20, valid_split=0.20) -> 'Tuple[DataSet, DataSet, DataSet]':
        return cls(xvec,yvec).split_data_valid_train_test(test_split=test_split, valid_split=valid_split)

    def split_data_valid_train_test(self, test_split=0.20, valid_split=0.20) -> 'Tuple[DataSet, DataSet, DataSet]':
        # split datasets into train/validation/test
        x_train, x_split, y_train, y_split = train_test_split(self.X, self.Y,
                                                              stratify=self.Y,
                                                              test_size=(test_split + valid_split))

        x_valid, x_test, y_valid, y_test = train_test_split(x_split, y_split,
                                                            stratify=y_split,
                                                            test_size=(test_split / (test_split + valid_split)))
        # put validation set back into training set
        x_train += x_valid
        y_train += y_valid

        # save these for return values
        train = self.__class__(X=x_train, Y=y_train, encoder=self.encoder)
        valid = self.__class__(X=x_valid, Y=y_valid, encoder=self.encoder)
        test = self.__class__(X=x_test, Y=y_test, encoder=self.encoder)
        return train, valid, test

    @classmethod
    def from_files(cls, code_file, example_file, target_level, combine=False, append_empty_class=False):
        level_codes = read_levels(code_file)
        title_records = read_titles(example_file, target_level)
        if combine:
            (xvec, yvec) = generate_combined(level_codes, title_records, target_level)
        else:
            (xvec, yvec) = generate_uncombined_text_for_target_level(title_records, target_level)

        if append_empty_class:
            append_empty_string_class(xvec, yvec)
        return cls(xvec, yvec)

    def save_as_pickle(self, file, is_path=False):
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'DataSet':
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        ds = pickle.load(handle)
        handle.close()
        return ds