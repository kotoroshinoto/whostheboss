#!/usr/bin/env bash
python -m scribe_classifier.nn_main model test_to_files \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl1.bayes.P 1 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl1.sgdsv.P 1 sgd \
--model nnmodels/ANN/neural_net_level1.P 1 ann \
--model source_data/pickles/canada/trained_models/simple.lvl2.bayes.P 2 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl2.sgdsv.P 2 sgd \
--model nnmodels/ANN/neural_net_level2.P 2 ann \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model nnmodels/ANN/neural_net_level3.P 3 ann \
--model source_data/pickles/canada/trained_models/simple.lvl4.bayes.P 4 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl4.sgdsv.P 4 sgd \
--model nnmodels/ANN/neural_net_level4.P 4 ann \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

python -m scribe_classifier.cli.main model ensemble test_predict_from_files \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model bayes 1 \
--model sgd 1 \
--model ann 1 \
--model bayes 2 \
--model sgd 2 \
--model ann 2 \
--model bayes 3 \
--model sgd 3 \
--model ann 3 \
--model bayes 4 \
--model sgd 4 \
--model ann 4 \
1

python -m scribe_classifier.cli.main model ensemble test_predict_from_files \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model bayes 1 \
--model sgd 1 \
--model ann 1 \
--model bayes 2 \
--model sgd 2 \
--model ann 2 \
--model bayes 3 \
--model sgd 3 \
--model ann 3 \
--model bayes 4 \
--model sgd 4 \
--model ann 4 \
2

python -m scribe_classifier.cli.main model ensemble test_predict_from_files \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model bayes 1 \
--model sgd 1 \
--model ann 1 \
--model bayes 2 \
--model sgd 2 \
--model ann 2 \
--model bayes 3 \
--model sgd 3 \
--model ann 3 \
--model bayes 4 \
--model sgd 4 \
--model ann 4 \
3

python -m scribe_classifier.cli.main model ensemble test_predict_from_files \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P \
--model bayes 1 \
--model sgd 1 \
--model ann 1 \
--model bayes 2 \
--model sgd 2 \
--model ann 2 \
--model bayes 3 \
--model sgd 3 \
--model ann 3 \
--model bayes 4 \
--model sgd 4 \
--model ann 4 \
4