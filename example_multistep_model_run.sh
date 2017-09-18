#!/usr/bin/env bash

#generate Training Sets
python \
./main.py canada_model dataset \
--example_file ./source_data/processed/NOC/all_examples \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl2.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl2.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl2.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 2

python \
./main.py canada_model dataset \
--example_file ./source_data/processed/NOC/all_examples \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl3.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl3.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 3

#train model
python ./main.py canada_model multi train \
--code_file ./source_data/processed/NOC/all_codes \
--model_filepath ./source_data/pickles/canada/trained_models/multi.lvl3.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--target_level 2 \
--emptyset NA

python ./main.py canada_model multi train \
--code_file ./source_data/processed/NOC/all_codes \
--model_filepath ./source_data/pickles/canada/trained_models/multi.lvl3.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--target_level 3 \
--emptyset NA

#test model
python ./main.py canada_model multi test --emptyset NA \
./source_data/pickles/canada/trained_models/multi.lvl3.P \
./source_data/pickles/canada/test_sets/valid.set.lvl3.P \
./source_data/pickles/canada/test_sets/test.set.lvl3.P \
3
