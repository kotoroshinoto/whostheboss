#!/usr/bin/env python
import pandas as pd
from scribe_classifier.data.canada.NOCdb.readers.titles import TitleSet
from sklearn import metrics

from scribe_classifier.data.canada.NOCdb.models import MultiStepModel

target_level = 3




preds = msm.predict(title_set=titles)

df = titles.to_dataframe(target_level=4)
df['preds'] = pd.Series(preds)
# print(df)

print(metrics.classification_report(titles.get_code_vec(target_level=target_level), df['preds']))