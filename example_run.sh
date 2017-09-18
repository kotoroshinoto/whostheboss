#!/usr/bin/env bash
#TODO need a way to pre-split the dataset so I can use the same validation/test sets on
#TODO both combined and uncombined versions to have a proper test of it

echo "Processing DataSet"
python \
./main.py canada_model gen_data \
--example_file ./TrainingData/training_sources/raw/NOC/all_examples \
--train_filepath ./Validation_And_Test_Sets/train.set.P \
--valid_filepath ./Validation_And_Test_Sets/valid.set.P \
--test_filepath ./Validation_And_Test_Sets/test.set.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target 2 \
--emptyset \
 || { echo "Processing Failed"; return; }
echo "Finished Processing"


echo "Starting Uncombined Train"
python \
./main.py canada_model simple \
--target 2 \
--train_filepath ./Validation_And_Test_Sets/train.set.P \
--model_filepath ./TrainedModels/simple.P \
 || { echo "Train Failed"; return; }
echo "Completed Uncombined Train"

echo "Starting Combined Train"
python \
./main.py canada_model simple \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
--target 2 \
--train_filepath ./Validation_And_Test_Sets/train.set.P \
--model_filepath ./TrainedModels/simple.combined.P \
 || { echo "Combined Train Failed"; return; }
echo "Completed Combined Train"

python \
./main.py canada_model test_simple TrainedModels/simple.P Validation_And_Test_Sets/valid.set.P Validation_And_Test_Sets/test.set.P 2

python \
./main.py canada_model test_simple \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
TrainedModels/simple.combined.P Validation_And_Test_Sets/valid.set.P Validation_And_Test_Sets/test.set.P 2

python \
./main.py canada_model test_simple --no-test_combine \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
TrainedModels/simple.combined.P Validation_And_Test_Sets/valid.set.P Validation_And_Test_Sets/test.set.P 2
