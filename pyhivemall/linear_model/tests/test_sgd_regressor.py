import unittest
from sklearn import linear_model

from pyhivemall.linear_model import SGDRegressor


class SGDRegressorTestCase(unittest.TestCase):

    def test_parse_options(self):
        opts = SGDRegressor.parse_options('')
        clf = SGDRegressor(**opts)
        self.assertTrue(isinstance(clf, linear_model.SGDRegressor))
