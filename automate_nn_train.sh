#!/usr/bin/env bash
python -m scribe_classifier.main canada_model_neural \
train --epoch 15 \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 1

python -m scribe_classifier.main canada_model_neural \
train --epoch 300 \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 2



python -m scribe_classifier.main canada_model_neural \
train --epoch 300 \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 3

