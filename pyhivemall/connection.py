from pyhive import hive
import pandas as pd
import pandas_td as td


class BaseConnection(object):

    def __init__(self, **kwargs):
        pass

    def fetch_table(self, table, **kwargs):
        return None


class HiveConnection(BaseConnection):

    def __init__(self, host='localhost', port=10000, database='default'):
        self.conn = hive.Connection(host=host, port=port, database=database)

    def fetch_table(self, table, columns=['*']):
        return pd.read_sql('select %s from %s' % (', '.join(columns), table), self.conn)


class TdConnection(BaseConnection):

    def __init__(self, apikey, endpoint, database='sample_datasets'):
        conn = td.connect(apikey=apikey, endpoint=endpoint)
        self.engine = td.create_engine('presto:{}'.format(database), conn)

    def fetch_table(self, table):
        return td.read_td_table(table, self.engine)
