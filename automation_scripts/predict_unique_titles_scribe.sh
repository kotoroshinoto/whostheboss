#!/usr/bin/env bash

python -m scribe_classifier.cli.main uniques generate_probability_matrices \
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
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--batchsize 25000 \
--keras_batch 512 \
--emptyset NA \
--basepath source_data/processed/scribe/unique_labels/proba_matrices \
source_data/processed/scribe/unique_labels/titles.txt