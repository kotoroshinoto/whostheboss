#!/usr/bin/env bash
python keras_classification.py train --epoch 5 --batch_size 16 --first_layer_size 2048 --layer 1048 1 True --layer 512 2 True --layer 256 1 True --layer 128 2 True 1

python keras_classification.py train --epoch 25 --batch_size 64 --first_layer_size 2048 --layer 2048 5 True --layer 1048 4 True --layer 512 3 True --layer 256 2 True 2

python keras_classification.py train --epoch 20  --layers 6 --layer_size=2048 --max_features 25000 3