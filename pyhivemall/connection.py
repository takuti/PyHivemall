from pyhive import hive
import pandas as pd
import pandas_td as td


class BaseConnection(object):

    def __init__(self, **kwargs):
        pass

    def fetch_table(self, table, **kwargs):
        return None

    def import_frame(self, frame, table):
        pass


class HiveConnection(BaseConnection):

    def __init__(self, host='localhost', port=10000, database='default'):
        self.conn = hive.Connection(host=host, port=port, database=database)

    def fetch_table(self, table, columns=['*']):
        return pd.read_sql('select %s from %s' % (', '.join(columns), table), self.conn)

    def import_frame(self, frame, table):
        # dataframe.to_sql(table, self.conn, if_exists='replace', index=False)
        raise NotImplementedError('For a DBAPI2 connection, pandas.DataFrame.to_sql does not support Hive. Maybe HiveConnection should be re-implemented with SQLAlchemy connection.')


class TdConnection(BaseConnection):

    def __init__(self, apikey, endpoint, database='sample_datasets'):
        self.conn = td.connect(apikey=apikey, endpoint=endpoint)
        self.database = database
        self.engine = td.create_engine('presto:{}'.format(database), self.conn)

    def fetch_table(self, table, **kwargs):
        return td.read_td_table(table, self.engine)

    def import_frame(self, frame, table):
        td.to_td(frame, self.database + '.' + table, self.conn, if_exists='replace', index=False)
