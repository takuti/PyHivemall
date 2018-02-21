from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np


class GeneralLearnerBase(BaseEstimator):

    def fit(self, X, y):
        """Fit model."""

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """

    @staticmethod
    def load(conn, table, feature_column='feature', weight_column='weight', bias_feature=None, **kwargs):
        df = conn.fetch_table(table)

        intercept = np.array([0.])  # (1,)
        coef = np.array([[]])  # (1, n_feature)

        vocabulary = {}
        feature_names = []

        j = 0
        for i, row in df.iterrows():
            feature, weight = row[feature_column], row[weight_column]

            if feature == bias_feature:
                intercept[0] = float(weight)
                continue

            coef = np.append(coef, [[weight]], axis=1)

            vocabulary[feature] = j
            j += 1
            feature_names.append(feature)

        vectorizer = DictVectorizer(separator='#')
        vectorizer.vocabulary_ = vocabulary
        vectorizer.feature_names_ = feature_names

        """
        lr = GeneralRegressor(**kwargs)

        lr.fit_intercept = bias_feature is not None

        lr.intercept_ = intercept
        lr.coef_ = coef

        return lr, vectorizer
        """

        return coef, intercept, vectorizer

    def store(self, conn, table, vocabulary, feature_column='feature', weight_column='weight', bias_feature=None):
        df = self._to_frame(vocabulary, feature_column, weight_column, bias_feature)
        conn.import_frame(df, table)

    def _to_frame(self, vocabulary, feature_column, weight_column, bias_feature):
        data = []

        for feature, index in vocabulary.items():
            data.append((feature, self.coef_[0, index]))

        if bias_feature is not None:
            data.append((bias_feature, self.intercept_[0]))

        return pd.DataFrame.from_records(data, columns=[feature_column, weight_column])
