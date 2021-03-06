#!/usr/bin/env bash

#creating and pickling datasets from canadian database pickles
python \
-m scribe_classifier.cli.main dataset pickle \
--example_file ./source_data/pickles/canada/tidy_sets/all_titles.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl1.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl1.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl1.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 1

python \
-m scribe_classifier.cli.main dataset pickle \
--example_file ./source_data/pickles/canada/tidy_sets/all_titles.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl2.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl2.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl2.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 2

python \
-m scribe_classifier.cli.main dataset pickle \
--example_file ./source_data/pickles/canada/tidy_sets/all_titles.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl3.P \
--valid_filepath ./source_data/pickles/canada/test_sets/valid.set.lvl3.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl3.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target_level 3

python \
-m scribe_classifier.cli.main dataset pickle \
--example_file ./source_data/pickles/canada/tidy_sets/all_titles.P \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--valid_prop 0.0 \
--test_prop 0.20 \
--target_level 4
