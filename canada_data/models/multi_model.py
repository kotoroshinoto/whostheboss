# predict 0-9 & Unknown @ first level, then internal classes at next level
from canada_data.readers.codes import AllCodes
from .simple_model import SimpleModel
from typing import Dict, List, Tuple
from canada_data.readers.titles import TitleSet


class MultiStepModel:
    def __init__(self, all_codes_filename, target_level=1):
        self.target_level = target_level
        self.all_codes = AllCodes()
        self.all_codes.add_codes_from_file(filename=all_codes_filename)
        self.models = dict()  # type: Dict['str', SimpleModel]
        self.codes_by_level = self.all_codes.get_codes_for_fitting_multi_level(target_level=self.target_level)
        for i in range(self.target_level):
            # print("%d" % i)
            for code in self.codes_by_level[i]:
                # print("\t'%s'" % code)
                # TODO detect when there is only 1 class at a given level for a given parent category and allow fitting/predicting to pass through
                self.models[code] = SimpleModel(target_level=(i+1))

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

