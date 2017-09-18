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
            for code in self.codes_by_level[i]:
                self.models[code] = SimpleModel()

    def fit(self, title_set: 'TitleSet'):
        sets_for_codes = title_set.get_sets_for_fitting_multi_level(
            all_codes=self.all_codes,
            target_level=self.target_level
        )
        for setkey in sets_for_codes:
            self.models[setkey].fit(sets_for_codes[setkey])

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

