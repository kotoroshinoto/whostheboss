#!/usr/bin/env bash

#getting prediction metrics
python -m scribe_classifier.cli.main model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl1.oversample.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
1

python -m scribe_classifier.cli.main model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl2.oversample.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
2

python -m scribe_classifier.cli.main model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl3.oversample.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
3

python -m scribe_classifier.cli.main model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl4.oversample.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
4
