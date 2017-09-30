import pickle


class ObjectPickler:
    @staticmethod
    def save_as_pickle(obj, filepath):
        pickle_file = open(filepath, 'wb')
        pickle.dump(obj, pickle_file)

    @staticmethod
    def load_from_pickle(filepath):
        pickle_file = open(filepath,'rb')
        return pickle.load(pickle_file)