#!/usr/bin/env bash

#training models
python \
-m scribe_classifier.main canada_model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl1.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl1.P \
--target_level 1 \
--emptyset NA \
--model_type 'sgdsv'

python \
-m scribe_classifier.main canada_model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl2.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl2.P \
--target_level 2 \
--emptyset NA \
--model_type 'sgdsv'

python \
-m scribe_classifier.main canada_model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl3.P \
--target_level 3 \
--emptyset NA \
--model_type 'sgdsv'


