#!/usr/bin/env bash

python -m scribe_classifier.cli.main uniques predict_from_matrices \
~/scribebinclustering/source_data/processed/scribe/unique_labels/titles.txt \
~/scribebinclustering/source_data/processed/scribe/unique_labels/titles_classified.tsv \
--emptyset NA \
--code_file source_data/pickles/canada/tidy_sets/all_codes.P \
--basepath ~/scribebinclustering/source_data/processed/scribe/unique_labels/proba_matrices \
--levels 1 1 \
--model ann 1
