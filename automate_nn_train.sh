#!/usr/bin/env bash
python keras_classification.py train --epoch 20 --layers 6 --layer_size=2048 --max_features 25000 1

python keras_classification.py train --epoch 20  --layers 6 --layer_size=2048 --max_features 25000 2

python keras_classification.py train --epoch 20  --layers 6 --layer_size=2048 --max_features 25000 3