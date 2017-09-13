import sqlalchemy
import sqlalchemy.orm
import pandas as pd
from kmodes import kmodes
from kmodes import kprototypes

class DB_handler:
    def __init__(self,user: str, password: str, url: str):
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


if __name__ == "__main__":
    user = "mgooch"
    pwd = "SSJ699Goku!"
    dbstr = "localhost/scribe"
    dbh = DB_handler(user, pwd, dbstr)
    df = dbh.grab_usa_medium_tech_data()
    print(df.shape)
    print(df.columns)
