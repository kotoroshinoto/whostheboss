#!/usr/bin/env python
import click
from scribe_classifier.data.scribe import TrainingSetMap, generate_training_set_defs_from_path, generate_training_set


@click.group()
def training_set_cli():
    pass


@training_set_cli.group()
def generate():
    pass


@generate.command(name='defs_from_path')
@click.option('--directory', '-d', type=click.Path(exists=True), required=True, help="directory containing training set defs in tree form")
@click.option('--deffile', '-f', type=click.File('wb'), required=True, help="file to write training set defs in pickle form")
def generate_pickle_defs(directory, deffile):
    generate_training_set_defs_from_path(directory, deffile)


@generate.command(name='set_and_defs')
@click.option('--directory', '-d', type=click.Path(exists=True), required=True, help="directory containing training set defs in tree form")
@click.option('--output', '-o', type=click.File('wb'),required=True, help="path to write generated training set")
@click.option('--source', '-s', type=click.File('r'), required=True,  help="source of strings to be used to generate set")
@click.option('--deffile', '-f', type=click.File('wb'), default=None, help="file to write training set defs in pickle form")
def generate_from_path(directory, output, source, deffile):
    tsetmap = generate_training_set_defs_from_path(directory, deffile)
    generate_training_set(tsetmap, source, output)


@generate.command(name='set_from_defs')
@click.option('--deffile', '-f', type=click.File('rb'), required=True, help="file containing training set defs in pickle form")
@click.option('--output', '-o', type=click.File('wb'), required=True, help="path to write generated training set")
@click.option('--source', '-s', type=click.File('r'), required=True, help="source of strings to be used to generate set")
def generate_from_file(deffile, output, source):
    tsetmap = TrainingSetMap.from_pickle(deffile)
    generate_training_set(tsetmap, source, output)


if __name__ == "__main__":
    training_set_cli()
