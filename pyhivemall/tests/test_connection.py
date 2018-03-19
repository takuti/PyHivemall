import unittest
import os
import time
import pandas as pd

from pyhivemall import HiveConnection, TdConnection


class HiveConnectionTestCase(unittest.TestCase):

    def setUp(self):
        try:
            self.conn = HiveConnection(host='localhost', port=10000, database='default')
        except Exception:
            raise unittest.SkipTest('falied to find hive://localhost:10000/default')

    def test(self):
        df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        table = 'test_{}'.format(int(time.time()))

        self.conn.import_frame(df, table)

        pd.testing.assert_frame_equal(self.conn.fetch_table(table), df)

        # tear down
        cursor = self.conn.conn.cursor()
        cursor.execute('drop table {}'.format(table))


class TdConnectionTestCase(unittest.TestCase):

    @unittest.skipUnless('TD_API_KEY' in os.environ, '$TD_API_KEY is not set')
    @unittest.skipUnless('TD_API_SERVER' in os.environ, '$TD_API_SERVER is not set')
    def setUp(self):
        self.conn = TdConnection(apikey=os.environ['TD_API_KEY'],
                                 endpoint=os.environ['TD_API_SERVER'],
                                 database='sample_datasets')

    def test(self):
        # TODO: test `import_frame` somehow
        df_www = self.conn.fetch_table('www_access')
        self.assertEqual(df_www.shape, (5000, 9))
