from sklearn import linear_model
import numpy as np

from .base import LinearModel


class LogisticRegression(LinearModel, linear_model.LogisticRegression):

    @staticmethod
    def load(conn, table, feature_column='feature', weight_column='weight', bias_feature=None, **kwargs):
        coef, intercept, vectorizer = super(LogisticRegression, LogisticRegression).load(
            conn, table, feature_column, weight_column, bias_feature)

        lr = LogisticRegression(**kwargs)

        lr.intercept_ = intercept
        lr.coef_ = coef
        lr.classes_ = np.array([0, 1])

        return lr, vectorizer
