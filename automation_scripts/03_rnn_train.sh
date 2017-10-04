#!/usr/bin/env bash
#25
python recurrent_neural_net.py \
--emptyset NA \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test_filepath ./source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model_filepath nnmodels/rnnclf_lvl1.P \
--epochs 5 \
--batch_size 256 \
--target_level 1 \
--warmstart