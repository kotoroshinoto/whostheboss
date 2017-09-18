import os
import pickle


class TrainingSetMap:
    def __init__(self):
        #these will be the columns
        self.features = list()  # type: List[str]
        #these will map features to a list of categories
        self.categories = dict()  # type: Dict[str, List[str]]
        #catdefs['job title']['feature'] -> associated with feature or not
        self.cat_defs = dict()  # type: Dict[str, set[str]]

    @classmethod
    def from_dir_tree(cls, dirpath) -> 'TrainingSetMap':
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
        return pickle.load(filehandle)

    def save_as_pickle(self, filehandle):
        pickle.dump(self, filehandle)


class TrainingSet:
    def __init__(self):
        self.strings
        pass

    @classmethod
    def from_source_using_defs(cls, defs: TrainingSetMap, sourcehandle) -> 'TrainingSet':
        pass

    def save_as_pickle(self, filehandle):
        pickle.dump(self, filehandle)

    @staticmethod
    def from_pickle(filehandle) -> 'TrainingSet':
        return pickle.load(filehandle)


def generate_training_set_defs_from_path(directory, deffile) -> TrainingSetMap:
    tsetmap = TrainingSetMap.from_dir_tree(directory)
    if deffile is not None:
        tsetmap.save_as_pickle(deffile)
    return tsetmap


def generate_training_set(tsetmap, sourcehandle, outhandle):
    stringset = set()
    for line in sourcehandle:
        stringset.add(line)