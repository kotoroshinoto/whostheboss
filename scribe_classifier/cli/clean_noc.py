#!/usr/bin/env python
import sys
import os
import click
from csv import reader
from scribe_classifier.data.NOCdb.readers import TitleRecord, TitleSet, CodeRecord, CodeSet, TitlePreprocessor


@click.group()
def db_tidy_main_cli():
    """Operations to clean and prepare NOC database for use"""
    pass


@db_tidy_main_cli.command(name='codes')
@click.option('--infile', '-i', type=click.File('r', encoding='iso-8859-1'), required=True, help="original raw original NOC code file")
@click.option('--outfile', '-o', type=click.File('w'), default=None, help="file to write mapped output to in tabular format")
@click.option('--pickle', '-p', type=click.File('wb'), default=None, help="file to write mapped output to in pickle format")
def clean_codes(infile, outfile, pickle):
    """This program cleans up the NOC code file, saving the resulting code set in a pickle format.\n
    User can redirect stdout if they desire to save a tabular copy of the same data."""
    if outfile is None:
        outfile = sys.stdout
    rdr = reader(infile)
    first_line = next(rdr)
    # print("\t".join([first_line[0], first_line[1], first_line[3], first_line[5]]))
    filestr = 'code_lvl_%d'
    all_codes = CodeSet()
    for entry in rdr:
        if entry[1].rstrip().lstrip() == "":
            continue
        code_l = CodeSet.parse_code_column(entry[0])
        for code_s in code_l:
            kept_values = [code_s, entry[1], entry[3], entry[5]]
            if pickle is not None:
                all_codes.add_code(CodeRecord(code=code_s, desc=entry[1]))
            print("\t".join(kept_values), file=outfile)
    if pickle is not None:
        all_codes.save_as_pickle(file=pickle, is_path=False)


@db_tidy_main_cli.command(name='titles')
@click.option('--infile', '-i', type=click.File('r', encoding='iso-8859-1'), required=True, help="original raw original NOC example file")
@click.option('--outfile', '-o', type=click.File('w'), default=None, help="file to write mapped output to in tabular format")
@click.option('--pickle', '-p', type=click.File('wb'), default=None, help="file to write mapped output to in pickle format")
def clean_titles(infile, outfile, pickle):
    """This program will clean NOC title examples, splitting dual-gender titles, condensing some prefixes
    and rips out non alphanumeric characters, leaving single whitespaces between words.\n
    It creates a pickle file containing the prepared titleset. \n
    User can redirect stdout to capture a tabular copy of the file as well"""
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
