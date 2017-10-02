#!/usr/bin/env bash

#training models
python \
-m scribe_classifier.simple_main model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl1.bayes.P \
--target_level 1 \
--emptyset NA \
--model_type 'bayes'

python \
-m scribe_classifier.simple_main model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl2.bayes.P \
--target_level 2 \
--emptyset NA \
--model_type 'bayes'

python \
-m scribe_classifier.simple_main model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl3.bayes.P \
--target_level 3 \
--emptyset NA \
--model_type 'bayes'

python \
-m scribe_classifier.simple_main model simple train \
--train_filepath ./source_data/pickles/canada/test_sets/train.set.lvl4.P \
--model_filepath ./source_data/pickles/canada/trained_models/simple.lvl4.bayes.P \
--target_level 4 \
--emptyset NA \
--model_type 'bayes'
