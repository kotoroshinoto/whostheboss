#!/usr/bin/env bash

python main.py clean_titles codes \
--infile source_data/raw/NOC/iso-8859-1/122372_Code.csv \
--outfile source_data/processed/NOC/all_codes \
--pickle source_data/pickles/canada/tidy_sets/all_codes.P

python main.py clean_titles titles \
--infile source_data/raw/NOC/iso-8859-1/122372_Element.csv \
--outfile source_data/processed/NOC/all_examples \
--pickle source_data/pickles/canada/tidy_sets/all_titles.P > prepped_examples.tsv
