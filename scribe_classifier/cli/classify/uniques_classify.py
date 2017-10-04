import click
from scribe_classifier.data.NOCdb.readers.titles import TitlePreprocessor as tp
from typing import List


class ScribeTitle:
    def __init__(self, title_str):
        self.orig_title = str(title_str)
        self.title = tp.preprocess_slugify(tp.preprocess_title_prefixes(title_str))
        self.codes = []

    def __str__(self):
        # output will be, Original input \t cleaned input \t predicted class1 \t predicted class2 \t predicted class3 \t predicted class4 etc
        return "\t".join([self.orig_title, self.title, "\t".join(self.codes)])


def get_classification_titles(stl: 'List[ScribeTitle]'):
    retl = []
    for st in stl:
        retl.append(st.title)
    return retl


@click.command()
@click.argument('input', type=click.File('r'))
@click.argument('output', type=click.File('w'))
@click.option('--levels', default=(1, 4), type=(click.IntRange(min=1, max=4), click.IntRange(min=1, max=4)), help="")
def classify_uniques_cli(input, output, levels):
    titles = []
    count = 0
    for line in input:
        line = line.rstrip()
        count += 1
        st = ScribeTitle(title_str=line)
        titles.append(st)

