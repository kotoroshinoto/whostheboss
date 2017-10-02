# predict 0-9 & Unknown @ first level, then internal classes at next level
import pickle
from typing import Dict, List

from scribe_classifier.data.canada.models.simple.simple_model import SimpleModel
from scribe_classifier.data.canada.readers import TitleSet, CodeSet, TitleRecord


class MultiStepPredictions:
    """MultiStepPredictions represents the result of predicting on all levels"""
    def __init__(self):
        self.preds = dict()  # type: List[List[str]]

    def add_preds(self, preds: 'List[str]', target_level=1):
        """adds predictions"""
        self.preds[target_level] = preds

    def add_pred(self, target_level: int, pred: str):
        """adds a single prediction at target level"""
        while target_level > len(self.preds):
            self.add_preds(list())
        self.preds[target_level].append(pred)

    def get_preds(self, target_level: int):
        """gets predictions for target level"""
        return self.preds[target_level]


class OneClassFakeModel:
    """This is a fake model that always predicts 1 class. Used for emptyset and for classes that only have 1 child, and
    thus do not actually need a model"""
    def __init__(self, target_level, class_label):
        self.oneclass = class_label
        self.target_level = target_level

    def fit(self, title_set: 'TitleSet') -> List[str]:
        """since this always predicts one thing, it doesn't have to learn anything"""
        pass

    def predict(self, title_set):
        """fill a list with an equivalent number of uniform predictions of the assigned class"""
        preds = []
        for item in title_set.records:
            preds.append(self.oneclass)
        return preds

    def predict_one(self, title_record: 'TitleRecord') -> str:
        """predicts just once, given its oneclass assignment"""
        return self.oneclass


class MultiStepModel:
    """This model's concept was to predict the levels sequentially, banking on the level 1 predictions being more
    accurate, and thus training all the lower level models only to discriminate between the sibling classes instead of
    all of the classes. This ended up not working as well as using a simple model, and I haven't updated it in quite
    some number of iterations of changes to the rest of the codebase. it should be assumed broken until someone
    goes over the code to check"""
    def __init__(self, all_codes_filename, target_level=1, emptyset_label: str=None):
        """initialize an instance of multistep model using all_codes, emptyset label, and a target level"""
        self.target_level = target_level
        self.all_codes = CodeSet()
        self.all_codes.add_codes_from_file(filename=all_codes_filename)
        self.models = dict()  # type: Dict['str', SimpleModel]
        # add fakemodel for emptyclasses if needed, this will allow "NA" results to iteratively receive "NA" again
        if emptyset_label is not None:
            if emptyset_label == "":
                self.emptyset_label = "NA"
            else:
                self.emptyset_label = emptyset_label
            self.models[self.emptyset_label] = OneClassFakeModel(target_level=0, class_label=self.emptyset_label)
        self.codes_by_level = self.all_codes.get_codes_for_fitting_multi_level(target_level=4)
        for i in range(self.target_level):
            # print("%d" % i)
            for code in self.codes_by_level[i]:
                numchild = self.all_codes.get_num_children(code)
                # print("\t'%s \t %d'" % (code, numchild))
                if numchild > 1:
                    self.models[code] = SimpleModel(target_level=(i+1), emptyset_label=self.emptyset_label)
                else:
                    #there is nothing to predict if there is only one child class, use Fake Model
                    self.models[code] = OneClassFakeModel(target_level=(i+1), class_label=self.all_codes.get_children(code)[0])

    def fit(self, title_set: 'TitleSet'):
        """fit all of the models on the given title set"""
        sets_for_codes = title_set.get_sets_for_fitting_multi_level(
            all_codes=self.all_codes,
            target_level=self.target_level
        )
        for setkey in sets_for_codes:
            # print(setkey)
            code_set = sets_for_codes[setkey]
            # print(sorted(list(set(code_set.get_code_vec()))))
            # print("model believes its target level is: %d" % self.models[setkey].target_level)
            self.models[setkey].fit_titleset(code_set)

    def predict(self, title_set: 'TitleSet') -> 'MultiStepPredictions':
        """make class predictions for a given titleset"""
        all_preds = MultiStepPredictions()
        for i in range(self.target_level):
            all_preds.add_preds(list())

        for title_record in title_set.records:
            model = self.models[""]
            pred = model.predict_titlerecord(title_record=title_record)
            for i in range(1, self.target_level):
                all_preds.add_pred(target_level=i, pred=pred)
                model = self.models[pred]  # type: SimpleModel
                pred = model.predict_titlerecord(title_record=title_record)
            all_preds.add_pred(target_level=self.target_level, pred=pred)
        return all_preds

    def save_as_pickle(self, file, is_path=False):
        """save model as a pickle"""
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'MultiStepModel':
        """load multi step model from pickle"""
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        smdl = pickle.load(handle)
        handle.close()
        return smdl

