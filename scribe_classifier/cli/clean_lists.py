import re
import sys
import os
import click


list_extractor = re.compile("^unique (.+):\[(.+)\]$")
split_point = re.compile(", u['\"]")


def process_file(input_dir, filename):
    """processes the file to clean it properly"""
    retval = []
    cityfile = open(os.path.join(input_dir, filename), 'r')
    line = cityfile.readline()
    listmatch = list_extractor.match(line)
    if not listmatch:
        return []
    listname = listmatch.group(1)
    # print("extracting %s data_scribe_unique" % listname)
    content = listmatch.group(2)
    split_obj = split_point.split(content)
    for item in split_obj:
        if item == 'None' or (item.startswith('u') and (item[1] == '"' or item[1] == "'")):
            retval.append(item)
        else:
            retval.append("u%s%s" % (item[-1], item))
    return retval


def clean_file(input_dir, output_dir, filename):
    """clean given file and place it into output directory with given filename"""
    filecontent = process_file(input_dir=input_dir, filename=filename)
    outfile = open(os.path.join(output_dir, filename) % filename, 'w')
    # outfile = sys.stdout
    print("\n".join(filecontent), file=outfile)
    outfile.close()


@click.group()
def uniques_cli():
    """operations involving unique-entry files from scribe"""
    pass


@uniques_cli.command(name='tidy')
@click.argument('input_dir', type=click.Path(file_okay=False, dir_okay=True, exists=True, resolve_path=True, readable=True), default=None)
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, exists=True, resolve_path=True, writable=True), default=None)
def clean_all_unique_files(input_dir, output_dir):
    """clean unique entry files provided by scribe"""
    cwd = os.path.abspath('.')
    if input_dir is None:
        input_dir = os.path.join(cwd, 'source_data', 'raw', 'scribe', 'unique_labels')
        print("input_dir defaulting to: %s" % input_dir)
    if output_dir is None:
        output_dir = os.path.join(cwd, 'source_data', 'processed', 'scribe', 'unique_labels')
        print("output_dir defaulting to: %s" % output_dir)

    clean_file(input_dir=input_dir, output_dir=output_dir, filename="cities.txt")
    clean_file(input_dir=input_dir, output_dir=output_dir, filename="countries.txt")
    clean_file(input_dir=input_dir, output_dir=output_dir, filename="industries.txt")
    clean_file(input_dir=input_dir, output_dir=output_dir, filename="titles.txt")


if __name__ == "__main__":
    clean_all_unique_files()

