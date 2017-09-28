#!/usr/bin/env bash

#getting prediction metrics
python -m scribe_classifier.simple_main canada_model simple test --emptyset NA ./source_data/pickles/canada/trained_models/simple.lvl1.bayes.P ./source_data/pickles/canada/test_sets/valid.set.lvl1.P ./source_data/pickles/canada/test_sets/test.set.lvl1.P 1

python -m scribe_classifier.simple_main canada_model simple test --emptyset NA ./source_data/pickles/canada/trained_models/simple.lvl2.bayes.P ./source_data/pickles/canada/test_sets/valid.set.lvl2.P ./source_data/pickles/canada/test_sets/test.set.lvl2.P 2

python -m scribe_classifier.simple_main canada_model simple test --emptyset NA ./source_data/pickles/canada/trained_models/simple.lvl3.bayes.P ./source_data/pickles/canada/test_sets/valid.set.lvl3.P ./source_data/pickles/canada/test_sets/test.set.lvl3.P 3
