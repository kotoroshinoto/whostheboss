#!/usr/bin/env bash

#level 1
python -m scribe_classifier.nn_main model test 1 \
--emptyset NA \
--model_filepath nnmodels/ANN/neural_net_level1.P \
--train_filepath source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 2
python -m scribe_classifier.nn_main model test 2 \
--emptyset NA \
--model_filepath nnmodels/ANN/neural_net_level2.P \
--train_filepath source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 3
python -m scribe_classifier.nn_main model test 3 \
--emptyset NA \
--model_filepath nnmodels/ANN/neural_net_level3.P \
--train_filepath source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 4
python -m scribe_classifier.nn_main model test 4 \
--emptyset NA \
--model_filepath nnmodels/ANN/neural_net_level4.P \
--train_filepath source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath source_data/pickles/canada/test_sets/test.set.lvl4.P
