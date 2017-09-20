#!/usr/bin/env python
import click
from csv import reader
from scribe_classifier.data.canada import TitleRecord, TitleSet, CodeRecord, AllCodes, TitlePreprocessor
import sys
import os


@click.group()
def db_tidy_main_cli():
    pass


@db_tidy_main_cli.command(name='codes')
@click.option('--infile', '-i', type=click.File('r', encoding='iso-8859-1'), required=True, help="original raw original NOC code file")
@click.option('--outfile', '-o', type=click.File('w'), default=None, help="file to write mapped output to in tabular format")
@click.option('--pickle', '-p', type=click.File('wb'), default=None, help="file to write mapped output to in pickle format")
def clean_codes(infile, outfile, pickle):
    if outfile is None:
        outfile = sys.stdout
    rdr = reader(infile)
    first_line = next(rdr)
    # print("\t".join([first_line[0], first_line[1], first_line[3], first_line[5]]))
    filestr = 'code_lvl_%d'
    all_codes = AllCodes()
    for entry in rdr:
        if entry[1].rstrip().lstrip() == "":
            continue
        kept_values = [entry[0], entry[1], entry[3], entry[5]]
        if pickle is not None:
            all_codes.add_code(CodeRecord(code=entry[0], desc=entry[1]))
        print("\t".join(kept_values), file=outfile)
    if pickle is not None:
        all_codes.save_as_pickle(file=pickle, is_path=False)


@db_tidy_main_cli.command(name='titles')
@click.option('--infile', '-i', type=click.File('r', encoding='iso-8859-1'), required=True, help="original raw original NOC example file")
@click.option('--outfile', '-o', type=click.File('w'), default=None, help="file to write mapped output to in tabular format")
@click.option('--pickle', '-p', type=click.File('wb'), default=None, help="file to write mapped output to in pickle format")
def clean_titles(infile, outfile, pickle):
    """this program takes output for jobtitles from clean_lists.py
    and rips out non alphanumeric characters, leaving single whitespaces between words\n
    It will then map the old value to the new values and write the Dict to a pickle file for later use"""
    if outfile is None:
        outfile = sys.stdout
    rdr = reader(infile)
    first_line = next(rdr)
    tset = TitleSet()
    for entry in rdr:
        if int(entry[3]) != 19:
            continue
        if entry[1].rstrip().lstrip() == "":
            continue
        kept_values = [entry[0], entry[1], entry[3], entry[4]]
        print("\t".join(kept_values), file=outfile)
        if pickle is not None:
            tset.add_title(TitleRecord(code=entry[0], title=entry[1]))

    if pickle is not None:
        tset = TitlePreprocessor.preprocess_titleset_split_genders(tset=tset)
        tset = TitlePreprocessor.preprocess_titleset_split_genders(tset=tset)
        # tset = TitlePreprocessor.preprocess_titleset_split_chief_officer(tset=tset)
        tset = TitlePreprocessor.preprocess_titleset_prefixes(tset=tset)
        tset = TitlePreprocessor.preprocess_slugify_titleset(tset=tset)
        for trecord in tset.records:
            print("%s\t%s" % (trecord.code, trecord.title))
        tset.save_as_pickle(file=pickle, is_path=False)


if __name__ == "__main__":
    db_tidy_main_cli()
