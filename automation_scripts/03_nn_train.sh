#!/usr/bin/env bash
#25
python -m scribe_classifier.nn_main model \
train \
--emptyset NA \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model_filepath nnmodels/ANN/neural_net_level1.P \
--epoch 25 \
--activation tanh \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 \
1

#30
python -m scribe_classifier.nn_main model \
train \
--emptyset NA \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model_filepath nnmodels/ANN/neural_net_level2.P \
--epoch 30 \
--activation tanh \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 \
2

#40
python -m scribe_classifier.nn_main model \
train \
--emptyset NA \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model_filepath nnmodels/ANN/neural_net_level3.P \
--epoch 40 \
--activation tanh \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 \
3

#30
python -m scribe_classifier.nn_main model \
train \
--emptyset NA \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model_filepath nnmodels/ANN/neural_net_level4.P \
--epoch 30 \
--activation tanh \
--first_layer_size 2048 \
--layer 1024 4 0.5 \
--layer 768 2 0.5 \
--batch_size 256 \
4
