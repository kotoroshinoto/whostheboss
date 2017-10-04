import click
from scribe_classifier.data.NOCdb.readers.titles import TitlePreprocessor as tp
from scribe_classifier.data.NOCdb.models.ensemble import CombinedModels
from typing import List


class ScribeTitle:
    def __init__(self, title_str):
        self.orig_title = str(title_str)
        self.title = tp.preprocess_slugify(tp.preprocess_title_prefixes(title_str))
        self.codes = dict()

    def __str__(self):
        # output will be, Original input \t cleaned input \t predicted class1 \t predicted class2 \t predicted class3 \t predicted class4 etc
        codes = []
        for i in range(1,5):
            if i in self.codes:
                codes.append(self.codes[i])
        return "\t".join([self.orig_title, self.title, "\t".join(codes)])


def get_classification_titles(stl: 'List[ScribeTitle]'):
    retl = []
    for st in stl:
        retl.append(st.title)
    return retl


@click.command()
@click.option('--code_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), required=True, help="This file should contain all codes and descriptions in tab-separated format. It will be used to understand how to stratify the models")
@click.option('--model',
              type=click.Tuple((click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
                    click.INT,
                    click.STRING)),
              multiple=True,
              help="provide a model to use, its type [sgd, bayes, neural], and specify its target level")
@click.option('--batchsize', type=click.INT, default=4000, help="number of titles to predict per batch")
@click.option('--keras_batch', type=click.INT, default=32, help="keras batch size to use during prediction")
@click.option('--levels', default=(1, 4), type=(click.IntRange(min=1, max=4), click.IntRange(min=1, max=4)), help="")
@click.option('--emptyset', type=click.STRING, default=None, help="account for this emptyset label the model was trained with as necessary")
@click.argument('input_filepath', type=click.File('r'))
@click.argument('output_filepath', type=click.File('w'))
def classify_uniques_cli(input_filepath, output_filepath, levels, code_file, model, batchsize, keras_batch, emptyset):
    if emptyset == "":
        emptyset = "NA"
    titles = []
    count = 0
    print("loading models into combined ensemble")
    mdl_paths = dict()
    for tup in model:
        if tup[1] not in mdl_paths:
            mdl_paths[tup[1]] = dict()
        mdl_paths[tup[1]][tup[2]] = tup[0]
    if 1 not in mdl_paths:
        mdl_paths[1] = None
    if 2 not in mdl_paths:
        mdl_paths[2] = None
    if 3 not in mdl_paths:
        mdl_paths[3] = None
    if 4 not in mdl_paths:
        mdl_paths[4] = None
    cmb_mdls = CombinedModels(lvl1_mdls=mdl_paths[1],
                              lvl2_mdls=mdl_paths[2],
                              lvl3_mdls=mdl_paths[3],
                              lvl4_mdls=mdl_paths[4],
                              all_codes=code_file,
                              emptyset_label=emptyset)
    print("ensemble complete")
    print("loading data")
    for line in input_filepath:
        line = line.rstrip()
        count += 1
        st = ScribeTitle(title_str=line)
        titles.append(st)
    print("data loading complete")
    preds = dict()
    print("getting vector of input titles")
    clean_titles = get_classification_titles(titles)
    print("vector obtained")
    print("performing classifications")
    for i in range(levels[0], levels[1] + 1):
        preds[i] = cmb_mdls.batched_predict(clean_titles,
                                           batch_size=batchsize,
                                           target_level=i,
                                           keras_batch_size=keras_batch)
    print("performing classifications")
    print("combining classifications")
    for i in range(len(preds)):
        for j in range(levels[0], levels[1] + 1):
            titles[i].codes[j] = preds[j][i]
    print("combining classifications completed")
    print("writing output file")
    for item in titles:
        print(item, file=output_filepath)
    print("Operation completed")
