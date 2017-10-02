#!/usr/bin/env bash

#level 1
python -m scribe_classifier.nn_main model combined_test 1 \
--emptyset NA \
--lowmem \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl1.bayes.P 1 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl1.sgdsv.P 1 sgd \
--model nnmodels/ANN/neural_net_level1.frozen.P 1 ann \
--model source_data/pickles/canada/trained_models/simple.lvl2.bayes.P 2 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl2.sgdsv.P 2 sgd \
--model nnmodels/ANN/neural_net_level2.frozen.P 2 ann \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model nnmodels/ANN/neural_net_level3.frozen.P 3 ann \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 2
python -m scribe_classifier.nn_main model combined_test 2 \
--emptyset NA \
--lowmem \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl2.bayes.P 2 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl2.sgdsv.P 2 sgd \
--model nnmodels/ANN/neural_net_level2.frozen.P 2 ann \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model nnmodels/ANN/neural_net_level3.frozen.P 3 ann \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 3
python -m scribe_classifier.nn_main model combined_test 3 \
--emptyset NA \
--lowmem \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model nnmodels/ANN/neural_net_level3.frozen.P 3 ann \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P
