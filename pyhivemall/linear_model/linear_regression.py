from sklearn import linear_model

from .base import LinearModel


class LinearRegression(LinearModel, linear_model.LinearRegression):

    @staticmethod
    def load(conn, table, feature_column='feature', weight_column='weight', bias_feature=None, **kwargs):
        coef, intercept, vectorizer = super(LinearRegression, LinearRegression).load(
            conn, table, feature_column, weight_column, bias_feature)

        lr = LinearRegression(**kwargs)

        lr.fit_intercept = bias_feature is not None

        lr.intercept_ = intercept
        lr.coef_ = coef

        return lr, vectorizer
