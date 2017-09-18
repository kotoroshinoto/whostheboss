from canada_data.readers.titles import TitleSet
from canada_data.readers.codes import AllCodes
from canada_data.models.multi_model import MultiStepModel
import pandas as pd


target_level = 1

titles = TitleSet()
titles.add_titles_from_file(filename="./TrainingData/training_sources/raw/NOC/all_examples")

msm = MultiStepModel(all_codes_filename="./TrainingData/training_sources/raw/NOC/all_codes", target_level=target_level)

msm.fit(title_set=titles)
preds = msm.predict(title_set=titles)

df = titles.to_dataframe(target_level=4)
df['preds'] = pd.Series(preds)
print(df)