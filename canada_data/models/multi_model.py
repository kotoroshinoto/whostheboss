# predict 0-9 & Unknown @ first level, then internal classes at next level
from canada_data.readers.codes import AllCodes
from .simple_model import SimpleModel
from typing import Dict, List, Tuple
from canada_data.readers.titles import TitleSet


class OneClassFakeModel:
    def __init__(self, target_level, class_label):
        self.oneclass = class_label
        self.target_level = target_level

    def fit(self, title_set: 'TitleSet') -> List[str]:
        pass

    def predict(self, title_set):
        preds = []
        for item in title_set.records:
            preds.append(self.oneclass)

    def predict_one(self, title_record: 'TitleRecord') -> str:
        return self.oneclass


class MultiStepModel:
    def __init__(self, all_codes_filename, target_level=1):
        self.target_level = target_level
        self.all_codes = AllCodes()
        self.all_codes.add_codes_from_file(filename=all_codes_filename)
        self.models = dict()  # type: Dict['str', SimpleModel]
        self.codes_by_level = self.all_codes.get_codes_for_fitting_multi_level(target_level=4)
        for i in range(self.target_level):
            # print("%d" % i)
            for code in self.codes_by_level[i]:
                numchild = self.all_codes.get_num_children(code)
                # print("\t'%s \t %d'" % (code, numchild))
                if numchild > 1:
                    self.models[code] = SimpleModel(target_level=(i+1))
                else:
                    self.models[code] = OneClassFakeModel(target_level=(i+1), class_label=self.all_codes.get_children(code)[0])

    def fit(self, title_set: 'TitleSet'):
        sets_for_codes = title_set.get_sets_for_fitting_multi_level(
            all_codes=self.all_codes,
            target_level=self.target_level
        )
        for setkey in sets_for_codes:
            print(setkey)
            code_set = sets_for_codes[setkey]
            print(sorted(list(set(code_set.get_code_vec()))))
            print("model believes its target level is: %d" % self.models[setkey].target_level)
            self.models[setkey].fit(code_set)

    def predict(self, title_set: 'TitleSet') -> List[str]:
        preds = []
        for title_record in title_set.records:
            model = self.models[""]
            pred = model.predict_one(title_record=title_record)
            for i in range(1, self.target_level):
                model = self.models[pred]  # type: SimpleModel
                pred = model.predict_one(title_record=title_record)
            preds.append(pred)
        print(preds)
        return preds

