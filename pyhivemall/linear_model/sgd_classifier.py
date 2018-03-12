from sklearn import linear_model
import numpy as np
import argparse

from .base import LinearModel
from .options.learning_rate import LearningRateOptions
from .options.loss_function import LossFunctionOptions
from .options.optimizer import OptimizerOptions
from .options.regularization import RegularizationOptions


class SGDClassifier(LinearModel, linear_model.SGDClassifier):

    @staticmethod
    def parse_options(options):
        parser = argparse.ArgumentParser(description='Hivemall options for GeneralClassifier')
        parser = LearningRateOptions.setup(parser)
        parser = LossFunctionOptions.setup(parser, classification=True)
        parser = OptimizerOptions.setup(parser)
        parser = RegularizationOptions.setup(parser)
        opts = vars(parser.parse_known_args(options.split())[0])
        SGDClassifier.validate_options(opts)
        if opts['loss'] == 'squared':
            opts['loss'] = 'squared_loss'
        if opts['learning_rate'] == 'fixed':
            opts['learning_rate'] = 'constant'
        elif opts['learning_rate'] == 'inverse':
            opts['learning_rate'] = 'invscaling'
        return opts

    @staticmethod
    def load(conn, table, feature_column='feature', weight_column='weight', bias_feature=None, options=''):
        coef, intercept, vectorizer = super(SGDClassifier, SGDClassifier).load(
            conn, table, feature_column, weight_column, bias_feature)

        opts = SGDClassifier.parse_options(options)
        clf = SGDClassifier(**opts)

        clf.intercept_ = intercept
        clf.coef_ = coef
        clf.classes_ = np.array([0, 1])

        return clf, vectorizer
