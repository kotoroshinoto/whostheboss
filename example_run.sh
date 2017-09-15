#!/usr/bin/env bash
#TODO need a way to pre-split the dataset so I can use the same validation/test sets on
#TODO both combined and uncombined versions to have a proper test of it

echo "Starting Uncombined Train"
./canada_model.py simple \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
--example_file ./TrainingData/training_sources/raw/NOC/all_examples \
--target 2 \
--valid_prop 0.20 \
--test_prop 0.20 \
--no-combine \
--model_filepath ./TrainedModels/simple.P \
--valid_filepath ./Validation_And_Test_Sets/simple.valid.set.P \
--test_filepath ./Validation_And_Test_Sets/simple.test.set.P \
--emptyset
echo "Completed Uncombined Train"

echo "Starting Combined Train"
./canada_model.py simple \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
--example_file ./TrainingData/training_sources/raw/NOC/all_examples \
--target 2 \
--valid_prop 0.20 \
--test_prop 0.20 \
--combine \
--model_filepath ./TrainedModels/simple.combined.P \
--valid_filepath ./Validation_And_Test_Sets/simple.valid.set.combined.P \
--test_filepath ./Validation_And_Test_Sets/simple.test.set.combined.P \
--emptyset
echo "Completed Combined Train"

./canada_model.py test_simple TrainedModels/simple.P Validation_And_Test_Sets/simple.valid.set.P Validation_And_Test_Sets/simple.test.set.P
./canada_model.py test_simple TrainedModels/simple.combined.P Validation_And_Test_Sets/simple.valid.set.combined.P Validation_And_Test_Sets/simple.test.set.combined.P

