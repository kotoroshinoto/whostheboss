#!/usr/bin/env bash

#getting prediction metrics
python -m scribe_classifier.simple_main canada_model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl1.svc.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
1

python -m scribe_classifier.simple_main canada_model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl2.svc.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
2

python -m scribe_classifier.simple_main canada_model simple test --emptyset NA \
./source_data/pickles/canada/trained_models/simple.lvl3.svc.P \
./source_data/pickles/canada/test_sets/train.set.lvl4.P \
./source_data/pickles/canada/test_sets/test.set.lvl4.P \
3
