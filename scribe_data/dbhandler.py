import pandas as pd
import sqlalchemy
import sqlalchemy.orm
import pickle

class DbHandler:
    def __init__(self, url: str, user: str, password: str = None):
        if password is None:
            self.db = sqlalchemy.create_engine("postgresql://" + user + "@" + url)
        else:
            self.db = sqlalchemy.create_engine("postgresql://" + user + ":" + password + "@" + url)
        self.sessionmkr = sqlalchemy.orm.sessionmaker(autoflush=False, bind=self.db)

    def grab_usa_medium_tech_data(self):
        querystr = "select * from email_list where \"companyCountry\" = 'United States' and \"industry\" in ('computer software','information technology and services,internet','marketing and advertising','internet') and \"employeeCount\" < 500 and \"emailError\" = False;"
        sess = self.sessionmkr()
        df = pd.read_sql_query(querystr, self.db)
        return df

    def grab_usa_data(self):
        querystr = "select * from email_list where \"companyCountry\" = 'United States' and \"emailError\" = False;"
        sess = self.sessionmkr()
        df = pd.read_sql_query(querystr, self.db)
        return df


class DataFramePickler:
    @staticmethod
    def save_as_pickle(df: 'pd.DataFrame', filepath):
        pickle_file = open(filepath, 'wb')
        pickle.dump(df, pickle_file)

    @staticmethod
    def load_from_pickle(filepath)->'pd.DataFrame':
        pickle_file = open(filepath,'rb')
        return pickle.load(pickle_file)
