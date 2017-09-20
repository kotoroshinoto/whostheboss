#!/usr/bin/env bash

#training models
python \
./main.py canada_model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl1.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl1.oversample.P \
--target_level 1 \
--emptyset NA \
--oversample

python \
./main.py canada_model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl2.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl2.oversample.P \
--target_level 2 \
--emptyset NA \
--oversample

python \
./main.py canada_model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl3.oversample.P \
--target_level 3 \
--emptyset NA \
--oversample
