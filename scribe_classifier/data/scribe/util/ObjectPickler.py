import pickle


class ObjectPickler:
    @staticmethod
    def save_as_pickle(df, filepath):
        pickle_file = open(filepath, 'wb')
        pickle.dump(df, pickle_file)

    @staticmethod
    def load_from_pickle(filepath):
        pickle_file = open(filepath,'rb')
        return pickle.load(pickle_file)