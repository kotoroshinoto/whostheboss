#!/usr/bin/env python
import click
from scribe_classifier.cli import canada_model_cli
from scribe_classifier.cli import db_tidy_main_cli
from scribe_classifier.cli import training_set_cli
from scribe_classifier.cli import pull_scribe_data_cli
from scribe_classifier.cli import keras_classifier_cli


@click.group()
def main():
    pass


main.add_command(cmd=canada_model_cli, name="canada_model")
main.add_command(cmd=pull_scribe_data_cli, name="pull_data")
main.add_command(cmd=db_tidy_main_cli, name="clean_titles")
main.add_command(cmd=training_set_cli, name="custom_training_set")
main.add_command(cmd=keras_classifier_cli, name='canada_model_neural')

if __name__ == "__main__":
    main()
