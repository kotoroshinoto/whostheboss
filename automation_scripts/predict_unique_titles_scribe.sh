#!/usr/bin/env bash

python -m scribe_classifier.classify_main uniques \--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl1.bayes.P 1 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl1.sgdsv.P 1 sgd \
--model nnmodels/ANN/neural_net_level1.P 1 neural \
--model source_data/pickles/canada/trained_models/simple.lvl2.bayes.P 2 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl2.sgdsv.P 2 sgd \
--model nnmodels/ANN/neural_net_level2.P 2 neural \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model nnmodels/ANN/neural_net_level3.P 3 neural \
--model source_data/pickles/canada/trained_models/simple.lvl4.bayes.P 4 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl4.sgdsv.P 4 sgd \
--model nnmodels/ANN/neural_net_level4.P 4 neural \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--batchsize 25000 \
--keras_batch 512 \
--levels 1 4 \
--emptyset NA \
source_data/processed/scribe/unique_labels/titles.txt \
source_data/processed/scribe/unique_labels/titles_classified.tsv
