#!/usr/bin/env bash
python -m scribe_classifier.nn_main canada_model \
train \
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
--batch_size 256 1

python -m scribe_classifier.nn_main canada_model \
freeze \
--model nnmodels/ANN/neural_net_level1.P \
--frozen source_data/pickles/canada/trained_models/ann/neural_net_level1.frozen.P

python -m scribe_classifier.nn_main canada_model \
train \
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
--batch_size 256 2

python -m scribe_classifier.nn_main canada_model \
freeze \
--model nnmodels/ANN/neural_net_level2.P \
--frozen source_data/pickles/canada/trained_models/ann/neural_net_level2.frozen.P

python -m scribe_classifier.nn_main canada_model \
train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model_filepath nnmodels/ANN/neural_net_level3.P \
--epoch 35 \
--activation tanh \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 3

python -m scribe_classifier.nn_main canada_model \
freeze \
--model nnmodels/ANN/neural_net_level3.P \
--frozen source_data/pickles/canada/trained_models/ann/neural_net_level3.frozen.P
