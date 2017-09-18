#!/usr/bin/env python
import click
from scribe_data.dbhandler import DbHandler
from scribe_data.dbhandler import DataFramePickler


@click.command()
@click.argument('url', type=click.STRING, required=True)
@click.option('--user', type=click.STRING, prompt=True, hide_input=False, required=True)
@click.option('--pwd', type=click.STRING, prompt=True, hide_input=True, required=True)
def pull_scribe_data_cli(url, user, pwd):
    """Pulls Midsize USA Tech from SQL"""
    dbh = DbHandler(url, user, pwd)
    df = dbh.grab_usa_medium_tech_data()
    DataFramePickler.save_as_pickle(df, './SavedScribeQueries/midsize_tech_usa.P')


if __name__ == "__main__":
    pull_scribe_data_cli()
