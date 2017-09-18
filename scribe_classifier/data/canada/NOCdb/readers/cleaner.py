#!/usr/bin/env python
import click
import codecs
from csv import reader


def convert_encoding(infile, outfile):
    old_encoding = 'iso-8859-1'
    desired_encoding = 'utf-8'
    BLOCKSIZE = 1048576
    with codecs.open(infile, 'r', old_encoding) as sourcefile:
        with codecs.open(outfile, 'w', desired_encoding) as destfile:
            while True:
                contents = sourcefile.read(BLOCKSIZE)
                if not contents:
                    break
                destfile.write(contents)


def clean_codes():
    handle = open('UTF8/122372_Code.csv', 'r')
    rdr = reader(handle)
    first_line = next(rdr)
    print("\t".join([first_line[0], first_line[1], first_line[3], first_line[5]]))
    filestr = 'code_lvl_%d'
    level_handles = []
    for i in range(1,5):
        print("%d" % i)
        level_handles.append(open(filestr % i,'w'))
    for entry in rdr:
        if entry[1].rstrip().lstrip() == "":
            continue
        kept_values=[entry[0], entry[1], entry[3], entry[5]]
        print("\t".join(kept_values), file=level_handles[int(entry[5]) - 1])
    handle.close()


def clean_elements():
    handle = open('UTF8/122372_Element.csv', 'r')
    rdr = reader(handle)
    first_line = next(rdr)
    print("\t".join([first_line[0], first_line[1], first_line[3], first_line[4]]))
    for entry in rdr:
        if int(entry[3]) != 19:
            continue
        if entry[1].rstrip().lstrip() == "":
            continue
        kept_values = [entry[0], entry[1], entry[3], entry[4]]
        print("\t".join(kept_values))
    handle.close()


@click.command()
def clean_canada_data():
    clean_codes()
    clean_elements()


if __name__ == '__main__':
    clean_canada_data()
