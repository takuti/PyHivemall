from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np

from .general_classifier import GeneralClassifier
from .general_regressor import GeneralRegressor
from .optimizer import SGD
from .optimizer import AdaGrad


class GeneralLearnerBase(BaseEstimator):

    def __init__(self, mini_batch=1, max_iter=100, disable_cv=False,
                 cv_rate=0.005, optimizer='adagrad', eps=1e-6, rho=0.95,
                 regularization='rda', l1_ratio=0.5, alpha=0.0001, eta='inverse',
                 eta0=0.1, total_steps=None, power_t=0.1, scale=100.0):

        self.mini_batch = mini_batch
        self.max_iter = max_iter
        self.disable_cv = disable_cv
        self.cv_rate = cv_rate

        optimizer = optimizer.lower()
        if optimizer == 'adagrad':
            self.optimizer = AdaGrad(regularization=regularization,
                                     l1_ratio=l1_ratio, alpha=alpha, eta=eta, eta0=eta0,
                                     total_steps=total_steps, power_t=power_t, eps=eps,
                                     scale=scale)
        elif optimizer == 'sgd':
            self.optimizer = SGD(regularization=regularization,
                                 l1_ratio=l1_ratio, alpha=alpha, eta=eta, eta0=eta0,
                                 total_steps=total_steps, power_t=power_t)
        elif optimizer == 'adadelta':
            raise NotImplementedError("optimizer 'adadelta' is not implemented yet")
        elif optimizer == 'adam':
            raise NotImplementedError("optimizer 'adam' is not implemented yet")
        else:
            raise ValueError("optimizer must be {'adagrad', 'sgd', 'adadelta', 'adam'}")

    def fit(self, X, y):
        """Fit model."""
        for i in range(self.max_iter):
            self.train(X[i], y[i])

    def update(self, x, y, predicted):
        # TODO: retain cumulative loss to check convergence
        # loss = self.loss_function.loss(predicted, y)

        dloss = self.loss_function.dloss(predicted, y)

        if self.mini_batch > 1:
            raise NotImplementedError('Mini-batch training is not implemented yet')

        for i in range(x.shape[0]):
            if x[i] == 0.0:
                continue
            x[i] = self.optimizer.update(None, self.coef_[i], dloss * x[i])

        self.optimizer.proceed_step()

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
        pred = np.dot(X, self.coef_.T)
        if self.fit_intercept:
            pred += self.intercept_.ravel()
        return pred

    @staticmethod
    def load(conn, table, feature_column='feature', weight_column='weight', bias_feature=None, classifier=True, regressor=False, **kwargs):
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

        if classifier:
            learner = GeneralClassifier(**kwargs)
        elif regressor:
            learner = GeneralRegressor(**kwargs)
        else:
            raise ValueError('Specify `regressor` or `classifier`')

        learner.fit_intercept = bias_feature is not None

        learner.intercept_ = intercept
        learner.coef_ = coef

        return learner, vectorizer

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
