import os
import pickle


# WARNING, this class is not completed and represented an early idea that was put to the side in favor of something else
class TrainingSetMap:
    """This object was intended to be used to help define custom buckets, it could still be retrofitted, but this avenue\n
    was not pursued"""
    def __init__(self):
        #these will be the columns
        self.features = list()  # type: List[str]
        #these will map features to a list of categories
        self.categories = dict()  # type: Dict[str, List[str]]
        #catdefs['job title']['feature'] -> associated with feature or not
        self.cat_defs = dict()  # type: Dict[str, set[str]]

    @classmethod
    def from_dir_tree(cls, dirpath) -> 'TrainingSetMap':
        """build map from directory tree"""
        newset = cls()  # type: TrainingSetMap
        topdir = os.path.abspath(dirpath)
        if not os.path.isdir(topdir):
            raise FileNotFoundError("dirpath is not a valid directory path")
        newset.features = [f.name for f in os.scandir(topdir) if f.is_dir() ]
        for feature in newset.features:
            fpath = os.path.join(topdir, feature)
            newset.categories[feature] = [f.name for f in os.scandir(fpath) if not f.is_dir()]
            for catlabel in newset.categories[feature]:
                catfilepath = os.path.join(fpath, catlabel)
                catfile = open(catfilepath, 'r')
                for line in catfile:
                    if len(line) == 0:
                        continue
                    if line not in newset.cat_defs:
                        newset.cat_defs[line] = set()
                    newset.cat_defs[line].add(catlabel)
                catfile.close()
        return newset

    @staticmethod
    def from_pickle(filehandle) -> 'TrainingSetMap':
        """load from pickle"""
        return pickle.load(filehandle)

    def save_as_pickle(self, filehandle):
        """save to pickle"""
        pickle.dump(self, filehandle)


class TrainingSet:
    """This class was supposed to represent all the titles within a given bucket"""
    def __init__(self):
        self.strings = []
        pass

    @classmethod
    def from_source_using_defs(cls, defs: TrainingSetMap, sourcehandle) -> 'TrainingSet':
        """This was going to read a source to produce the content"""
        pass

    def save_as_pickle(self, filehandle):
        """This was to save the object as a pickle"""
        pickle.dump(self, filehandle)

    @staticmethod
    def from_pickle(filehandle) -> 'TrainingSet':
        """this would reload the object from a pickle"""
        return pickle.load(filehandle)


def generate_training_set_defs_from_path(directory, deffile) -> TrainingSetMap:
    """This function would look at a directory tree and create a set of defs from the files inside"""
    tsetmap = TrainingSetMap.from_dir_tree(directory)
    if deffile is not None:
        tsetmap.save_as_pickle(deffile)
    return tsetmap


def generate_training_set(tsetmap, sourcehandle, outhandle):
    """This would make a training set given input source, output handle and a training set map"""
    stringset = set()
    for line in sourcehandle:
        stringset.add(line)