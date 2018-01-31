from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
import numpy as np


class LogisticRegression(linear_model.LogisticRegression):

    @staticmethod
    def load(source_dataframe=None, feature_column='feature', weight_column='weight', bias_feature=None, **kwargs):
        lr = linear_model.LogisticRegression(**kwargs)

        intercept = np.array([0.])  # (1,)
        coef = np.array([[]])  # (1, n_feature)

        vocabulary = {}
        feature_names = []

        if source_dataframe is not None:
            for i, row in source_dataframe.iterrows():
                feature, weight = row[feature_column], row[weight_column]

                if feature == bias_feature:
                    intercept[0] = float(weight)
                    continue

                coef = np.append(coef, [[weight]], axis=1)

                vocabulary[feature] = i
                feature_names.append(feature)

        lr.intercept_ = intercept
        lr.coef_ = coef
        lr.classes_ = np.array([0, 1])

        vectorizer = DictVectorizer(separator='#')
        vectorizer.vocabulary_ = vocabulary
        vectorizer.feature_names_ = feature_names

        return lr, vectorizer
