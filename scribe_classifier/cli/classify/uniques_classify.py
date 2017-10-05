import click
from scribe_classifier.data.NOCdb.readers.titles import TitlePreprocessor as tp
from scribe_classifier.data.NOCdb.models.ensemble import CombinedModels
from typing import List


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
    titles = []
    slugtitles = []
    for line in input_filepath:
        line = line.rstrip()
        count += 1
        titles.append(line)
        slugtitles.append(tp.preprocess_slugify(tp.preprocess_title_prefixes(line)))
    print("data loading complete")
    preds = dict()
    print("performing classifications")
    for i in range(levels[0], levels[1] + 1):
        preds[i] = cmb_mdls.batched_predict(slugtitles,
                                           batch_size=batchsize,
                                           target_level=i,
                                           keras_batch_size=keras_batch)
    print("performing classifications")
    print("writing output file")
    for j in range(len(preds)):
        text = [titles[j], slugtitles[j]]
        for i in range(levels[0], levels[1] + 1):
            text.append(preds[i][j])
        print("\t".join(text), file=output_filepath)
    print("Operation completed")

