from pyhive import hive
from sqlalchemy.engine import create_engine
import numpy as np
import pandas as pd
import pandas_td as td


class BaseConnection(object):

    def __init__(self, **kwargs):
        pass

    def fetch_table(self, table):
        return None

    def import_frame(self, frame, table):
        pass


class HiveConnection(BaseConnection):

    def __init__(self, host='localhost', port=10000, database='default'):
        self.conn = hive.Connection(host=host, port=port, database=database)
        self.engine = create_engine('hive://{}:{}/{}'.format(host, port, database))

    def fetch_table(self, table):
        return pd.read_sql_table(table, self.engine)

    def import_frame(self, frame, table):
        # NOTE: cannot fully utilize sqlalchemy due to a bug: https://github.com/dropbox/PyHive/issues/50
        # >>> frame.to_sql(table, self.engine, if_exists='replace', index=False, chunksize=10000)

        # create empty table with appropriate schema for given DataFrame
        frame.head(0).to_sql(table, self.engine, if_exists='replace', index=False)

        # build INSERT INTO statement and insert records via hive.Connection, not sqlalchemy engine
        # put single quotes before/after value of non-number column
        record = '(' + ', '.join(map(lambda c: ('%({})s' if frame[c].dtype == np.number else "'%({})s'").format(c), frame.columns)) + ')'
        values = []
        for _, row in frame.iterrows():
            values.append(record % row)

        cursor = self.conn.cursor()
        cursor.execute('insert into table {} values {}'.format(table, ', '.join(values)))


class TdConnection(BaseConnection):

    def __init__(self, apikey, endpoint, database='sample_datasets'):
        self.conn = td.connect(apikey=apikey, endpoint=endpoint)
        self.database = database
        self.engine = td.create_engine('presto:{}'.format(database), self.conn)

    def fetch_table(self, table):
        return td.read_td_table(table, self.engine)

    def import_frame(self, frame, table):
        td.to_td(frame, self.database + '.' + table, self.conn, if_exists='replace', index=False)
