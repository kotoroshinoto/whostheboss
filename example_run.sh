#!/usr/bin/env bash

python \
./main.py canada_model gen_data \
--example_file ./TrainingData/training_sources/raw/NOC/all_examples \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl1.P \
--valid_filepath ./Validation_And_Test_Sets/valid.set.lvl1.P \
--test_filepath ./Validation_And_Test_Sets/test.set.lvl1.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target 1 \
--emptyset

python \
./main.py canada_model simple \
--target 1 \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl1.P \
--model_filepath ./TrainedModels/simple.lvl1.P

./main.py canada_model test_simple TrainedModels/simple.lvl1.P Validation_And_Test_Sets/valid.set.lvl1.P Validation_And_Test_Sets/test.set.lvl1.P 1

python \
./main.py canada_model gen_data \
--example_file ./TrainingData/training_sources/raw/NOC/all_examples \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl2.P \
--valid_filepath ./Validation_And_Test_Sets/valid.set.lvl2.P \
--test_filepath ./Validation_And_Test_Sets/test.set.lvl2.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target 2 \
--emptyset

python \
./main.py canada_model simple \
--target 2 \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl2.P \
--model_filepath ./TrainedModels/simple.lvl2.P

python \
./main.py canada_model test_simple TrainedModels/simple.lvl2.P Validation_And_Test_Sets/valid.set.lvl2.P Validation_And_Test_Sets/test.set.lvl2.P 2

python \
./main.py canada_model gen_data \
--example_file ./TrainingData/training_sources/raw/NOC/all_examples \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl3.P \
--valid_filepath ./Validation_And_Test_Sets/valid.set.lvl3.P \
--test_filepath ./Validation_And_Test_Sets/test.set.lvl3.P \
--valid_prop 0.20 \
--test_prop 0.20 \
--target 3 \
--emptyset

python \
./main.py canada_model simple \
--target 3 \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl3.P \
--model_filepath ./TrainedModels/simple.lvl3.P

./main.py canada_model test_simple TrainedModels/simple.lvl3.P Validation_And_Test_Sets/valid.set.lvl3.P Validation_And_Test_Sets/test.set.lvl3.P 3

python \
./main.py canada_model simple \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
--target 2 \
--train_filepath ./Validation_And_Test_Sets/train.set.lvl2.P \
--model_filepath ./TrainedModels/simple.combined.lvl2.P


python \
./main.py canada_model test_simple \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
TrainedModels/simple.combined.lvl2.P Validation_And_Test_Sets/valid.set.lvl2.P Validation_And_Test_Sets/test.set.lvl2.P 2

python \
./main.py canada_model test_simple --no-test_combine \
--code_file ./TrainingData/training_sources/raw/NOC/all_codes \
TrainedModels/simple.combined.lvl2.P Validation_And_Test_Sets/valid.set.lvl2.P Validation_And_Test_Sets/test.set.lvl2.P 2