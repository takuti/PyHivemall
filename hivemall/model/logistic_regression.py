from sklearn import linear_model
import numpy as np


class LogisticRegression(linear_model.LogisticRegression):

    def __init__(self, source_dataframe=None, feature_column='feature', weight_column='weight', bias_feature=None, **kwargs):
        super(linear_model.LogisticRegression, self).__init__(**kwargs)

        intercept_ = np.array([0])
        coef_ = np.array([])

        if source_dataframe is not None:
            for i, row in source_dataframe.iterrows():
                if row[feature_column] == bias_feature:
                    intercept_[0] = float(row[weight_column])
                    continue
                # TODO: update coef_ here

        self.intercept_ = intercept_
        self.coef_ = coef_
