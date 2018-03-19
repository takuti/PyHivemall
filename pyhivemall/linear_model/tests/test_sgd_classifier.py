import unittest
from sklearn import linear_model

from pyhivemall.linear_model import SGDClassifier


class SGDClassifierTestCase(unittest.TestCase):

    def test_parse_options(self):
        opts = SGDClassifier.parse_options('')
        clf = SGDClassifier(**opts)
        self.assertTrue(isinstance(clf, linear_model.SGDClassifier))
