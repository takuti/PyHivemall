from sklearn import linear_model
import argparse

from .base import LinearModel
from .options.learning_rate import LearningRateOptions
from .options.loss_function import LossFunctionOptions
from .options.optimizer import OptimizerOptions
from .options.regularization import RegularizationOptions


class SGDRegressor(LinearModel, linear_model.SGDRegressor):

    @staticmethod
    def parse_options(options):
        parser = argparse.ArgumentParser(description='Hivemall options for GeneralRegressor')
        parser = LearningRateOptions.setup(parser)
        parser = LossFunctionOptions.setup(parser)
        parser = OptimizerOptions.setup(parser)
        parser = RegularizationOptions.setup(parser)
        opts = vars(parser.parse_known_args(options.split())[0])
        SGDRegressor.validate_options(opts)
        if opts['loss'] == 'squared':
            opts['loss'] = 'squared_loss'
        if opts['learning_rate'] == 'fixed':
            opts['learning_rate'] = 'constant'
        elif opts['learning_rate'] == 'inverse':
            opts['learning_rate'] = 'invscaling'
        return opts

    @staticmethod
    def load(conn, table, feature_column='feature', weight_column='weight', bias_feature=None, options=''):
        coef, intercept, vectorizer = super(SGDRegressor, SGDRegressor).load(
            conn, table, feature_column, weight_column, bias_feature)

        opts = SGDRegressor.parse_options(options)
        clf = SGDRegressor(**opts)

        clf.fit_intercept = bias_feature is not None

        clf.intercept_ = intercept
        clf.coef_ = coef

        return clf, vectorizer
