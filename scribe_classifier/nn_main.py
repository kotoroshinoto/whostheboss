#!/usr/bin/env python
import click
from scribe_classifier.cli.keras_classification import keras_classifier_cli


@click.group()
def nn_main():
    """Any models that use keras / tensorflow should be called from here"""
    pass


nn_main.add_command(cmd=keras_classifier_cli, name='canada_model')

if __name__ == "__main__":
    nn_main()
