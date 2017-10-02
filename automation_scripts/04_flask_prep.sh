#!/usr/bin/env bash
python flask_pregen.py reports --batch_size 25000 --keras_batch_size 128
python flask_pregen.py canada_plots --batch_size 25000 --keras_batch_size 128

python flask_pregen.py scribe_df --batch_size 25000 --keras_batch_size 128
python flask_pregen.py scribe_plots --batch_size 25000 --keras_batch_size 128