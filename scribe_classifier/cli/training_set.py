#!/usr/bin/env python
from typing import List, Dict
import sys
import os
import os.path
import click
import pickle
import fuzzywuzzy


class TrainingSetMap:
    def __init__(self):
        #these will be the columns
        self.features = list()  # type: List[str]
        #these will map features to a list of categories
        self.categories = dict()  # type: Dict[str, List[str]]
        #catdefs['job title']['feature'] -> associated with feature or not
        self.cat_defs = dict()  # type: Dict[str, set[str]]

    @classmethod
    def from_dir_tree(cls, dirpath) -> 'TrainingSetMap':
        newset = cls()  # type: TrainingSetMap
        topdir = os.path.abspath(dirpath)
        if not os.path.isdir(topdir):
            raise FileNotFoundError("dirpath is not a valid directory path")
        newset.features = [f.name for f in os.scandir(topdir) if f.is_dir() ]
        for feature in newset.features:
            fpath = os.path.join(topdir, feature)
            newset.categories[feature] = [f.name for f in os.scandir(fpath) if not f.is_dir()]
            for catlabel in newset.categories[feature]:
                catfilepath = os.path.join(fpath, catlabel)
                catfile = open(catfilepath, 'r')
                for line in catfile:
                    if len(line) == 0:
                        continue
                    if line not in newset.cat_defs:
                        newset.cat_defs[line] = set()
                    newset.cat_defs[line].add(catlabel)
                catfile.close()
        return newset

    @staticmethod
    def from_pickle(filehandle) -> 'TrainingSetMap':
        return pickle.load(filehandle)

    def save_as_pickle(self, filehandle):
        pickle.dump(self, filehandle)


class TrainingSet:
    def __init__(self):
        self.strings
        pass

    @classmethod
    def from_source_using_defs(cls, defs: TrainingSetMap, sourcehandle) -> 'TrainingSet':
        pass

    def save_as_pickle(self, filehandle):
        pickle.dump(self, filehandle)

    @staticmethod
    def from_pickle(filehandle) -> 'TrainingSet':
        return pickle.load(filehandle)


def generate_training_set_defs_from_path(directory, deffile) -> TrainingSetMap:
    tsetmap = TrainingSetMap.from_dir_tree(directory)
    if deffile is not None:
        tsetmap.save_as_pickle(deffile)
    return tsetmap


def generate_training_set(tsetmap, sourcehandle, outhandle):
    stringset = set()
    for line in sourcehandle:
        stringset.add(line)


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
