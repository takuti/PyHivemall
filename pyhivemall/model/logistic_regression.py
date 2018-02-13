from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd


class LogisticRegression(linear_model.LogisticRegression):

    @staticmethod
    def load(source_dataframe=None, feature_column='feature', weight_column='weight', bias_feature=None, **kwargs):
        lr = LogisticRegression(**kwargs)

        intercept = np.array([0.])  # (1,)
        coef = np.array([[]])  # (1, n_feature)

        vocabulary = {}
        feature_names = []

        if source_dataframe is not None:
            j = 0
            for i, row in source_dataframe.iterrows():
                feature, weight = row[feature_column], row[weight_column]

                if feature == bias_feature:
                    intercept[0] = float(weight)
                    continue

                coef = np.append(coef, [[weight]], axis=1)

                vocabulary[feature] = j
                j += 1
                feature_names.append(feature)

        lr.intercept_ = intercept
        lr.coef_ = coef
        lr.classes_ = np.array([0, 1])

        vectorizer = DictVectorizer(separator='#')
        vectorizer.vocabulary_ = vocabulary
        vectorizer.feature_names_ = feature_names

        return lr, vectorizer

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
