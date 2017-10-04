import click
from scribe_classifier.cli.classify.uniques_classify import classify_uniques_cli


@click.group(name="classify")
def classify():
    pass


classify.add_command(cmd=classify_uniques_cli, name='uniques')

if __name__ == "__main__":
    classify()
