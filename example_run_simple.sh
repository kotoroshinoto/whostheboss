#!/usr/bin/env bash

#creating and pickling datasets from canadian database
python \
./main.py canada_model gen_data \
--example_file ./source_data/processed/NOC/all_examples \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl1.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl1.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl1.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 1

python \
./main.py canada_model gen_data \
--example_file ./source_data/processed/NOC/all_examples \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl2.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl2.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl2.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 2

python \
./main.py canada_model gen_data \
--example_file ./source_data/processed/NOC/all_examples \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl3.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl3.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 3

#training models
python \
./main.py canada_model simple \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl1.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl1.P \
--target_level 1 \
--emptyset NA

python \
./main.py canada_model simple \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl2.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl2.P \
--target_level 2 \
--emptyset NA

python \
./main.py canada_model simple \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl3.P \
--target_level 3 \
--emptyset NA

#getting prediction metrics
python ./main.py canada_model test_simple --emptyset NA ./source_data/pickles/canada/trained_models/simple.lvl1.P ./source_data/pickles/canada/test_sets/valid.set.lvl1.P ./source_data/pickles/canada/test_sets/test.set.lvl1.P 1

python ./main.py canada_model test_simple --emptyset NA ./source_data/pickles/canada/trained_models/simple.lvl2.P ./source_data/pickles/canada/test_sets/valid.set.lvl2.P ./source_data/pickles/canada/test_sets/test.set.lvl2.P 2

python ./main.py canada_model test_simple --emptyset NA ./source_data/pickles/canada/trained_models/simple.lvl3.P ./source_data/pickles/canada/test_sets/valid.set.lvl3.P ./source_data/pickles/canada/test_sets/test.set.lvl3.P 3
