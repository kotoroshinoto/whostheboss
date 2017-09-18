#!/usr/bin/env python
import pandas as pd
from scribe_classifier.data.canada.NOCdb.readers.titles import TitleSet
from sklearn import metrics

from scribe_classifier.data.canada.NOCdb.models import MultiStepModel

target_level = 3

titles = TitleSet()
titles.add_titles_from_file(filename="./TrainingData/training_sources/raw/NOC/all_examples")

msm = MultiStepModel(all_codes_filename="./TrainingData/training_sources/raw/NOC/all_codes", target_level=target_level)
# exit()
msm.fit(title_set=titles)
preds = msm.predict(title_set=titles)

df = titles.to_dataframe(target_level=4)
df['preds'] = pd.Series(preds)
# print(df)

print(metrics.classification_report(titles.get_code_vec(target_level=target_level), df['preds']))