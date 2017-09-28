#!/usr/bin/env python
import click
from scribe_classifier.cli import canada_model_cli
from scribe_classifier.cli import db_tidy_main_cli
from scribe_classifier.cli import training_set_cli
from scribe_classifier.cli import pull_scribe_data_cli


@click.group()
def simple_main():
    """Data Cleaning / Setup methods can be called from here.\n
    Standard Sklearn models will be built from here"""
    pass


simple_main.add_command(cmd=canada_model_cli, name="canada_model")
simple_main.add_command(cmd=pull_scribe_data_cli, name="pull_data")
simple_main.add_command(cmd=db_tidy_main_cli, name="clean_titles")
simple_main.add_command(cmd=training_set_cli, name="custom_training_set")

if __name__ == "__main__":
    simple_main()
