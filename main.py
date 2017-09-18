#!/usr/bin/env python
import click
from cli.canada_model import canada_model_cli
from cli.pull_scribe_data import pull_scribe_data_cli
from cli.job_title_cleaner import job_title_cleaner_cli
from cli.training_set import training_set_cli


@click.group()
def main():
    pass

main.add_command(cmd=canada_model_cli, name="canada_model")
main.add_command(cmd=pull_scribe_data_cli, name="pull_data")
main.add_command(cmd=job_title_cleaner_cli, name="clean_titles")
main.add_command(cmd=training_set_cli, name="custom_training_set")

if __name__ == "__main__":
    main()
