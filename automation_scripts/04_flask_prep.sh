#!/usr/bin/env bash
python flask_pregen.py reports --batch_size 15000 --keras_batch_size 128
python flask_pregen.py canada_plots --force

python flask_pregen.py scribe_df --batch_size 15000 --keras_batch_size 128
python flask_pregen.py scribe_plots --force
