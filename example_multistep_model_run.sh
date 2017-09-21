#!/usr/bin/env bash

#train model
python -m scribe_classifier.main canada_model multi train \
--code_file ./source_data/processed/NOC/all_codes \
--model_filepath ./source_data/pickles/canada/trained_models/multi.lvl3.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--target_level 2 \
--emptyset NA
#--oversample

python -m scribe_classifier.main canada_model multi train \
--code_file ./source_data/processed/NOC/all_codes \
--model_filepath ./source_data/pickles/canada/trained_models/multi.lvl3.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--target_level 3 \
--emptyset NA
#--oversample

#test model
python -m scribe_classifier.main canada_model multi test --emptyset NA \
./source_data/pickles/canada/trained_models/multi.lvl3.P \
./source_data/pickles/canada/test_sets/valid.set.lvl3.P \
./source_data/pickles/canada/test_sets/test.set.lvl3.P \
3
