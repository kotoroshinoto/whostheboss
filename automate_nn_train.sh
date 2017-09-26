#!/usr/bin/env bash
python keras_classification.py train --epoch 15 \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 1

python keras_classification.py train --epoch 300 \
--first_layer_size 2048 \
--layer 1024 1 0.5 \
--layer 512 2 0.5 \
--layer 256 1 0.5 \
--layer 128 2 0.5 \
--batch_size 256 2



python keras_classification.py train --epoch 20  --layers 6 --layer_size=2048 --max_features 25000 \
3