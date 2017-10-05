#!/usr/bin/env bash

#level 1
python -m scribe_classifier.cli.main model ensemble test 1 \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl1.bayes.P 1 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl1.sgdsv.P 1 sgd \
--model source_data/pickles/canada/trained_models/simple.lvl2.bayes.P 2 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl2.sgdsv.P 2 sgd \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model source_data/pickles/canada/trained_models/simple.lvl4.bayes.P 4 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl4.sgdsv.P 4 sgd \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 2
python -m scribe_classifier.cli.main model ensemble test 2 \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl2.bayes.P 2 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl2.sgdsv.P 2 sgd \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model source_data/pickles/canada/trained_models/simple.lvl4.bayes.P 4 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl4.sgdsv.P 4 sgd \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 3
python -m scribe_classifier.cli.main model ensemble test 3 \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl3.bayes.P 3 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl3.sgdsv.P 3 sgd \
--model source_data/pickles/canada/trained_models/simple.lvl4.bayes.P 4 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl4.sgdsv.P 4 sgd \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P

#level 4
python -m scribe_classifier.cli.main model ensemble test 4 \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--model source_data/pickles/canada/trained_models/simple.lvl4.bayes.P 4 bayes \
--model source_data/pickles/canada/trained_models/simple.lvl4.sgdsv.P 4 sgd \
--val source_data/pickles/canada/test_sets/train.set.lvl4.P \
--test source_data/pickles/canada/test_sets/test.set.lvl4.P
