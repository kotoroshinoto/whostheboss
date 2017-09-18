#!/usr/bin/env python
import click


def clean_job_titles(inputhandle, outputhandle):
    pass


@click.command()
@click.option('--input', '-i', type=click.File('r'), required=True, help="file where each line is a jobtitle cleaned by clean_lists.py")
@click.option('--output', '-o', type=click.File('wb'), required=True, help="file to write mapped output to in pickle format")
def main(inputfile, outputfile):
    """this program takes output for jobtitles from clean_lists.py
    and rips out non alphanumeric characters, leaving single whitespaces between words\n
    It will then map the old value to the new values and write the Dict to a pickle file for later use"""
    pass


if __name__ == "__main__":
    main()