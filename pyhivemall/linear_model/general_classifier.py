from sklearn.base import ClassifierMixin
from sklearn.linear_model.sgd_fast import Hinge
from sklearn.linear_model.sgd_fast import Log
from sklearn.linear_model.sgd_fast import SquaredHinge
from sklearn.linear_model.sgd_fast import ModifiedHuber

from .general_learner_base import GeneralLearnerBase
from .general_regressor import GeneralRegressor


class GeneralClassifier(GeneralLearnerBase, ClassifierMixin):

    """Classification using generic optimizer

    Parameters
    ----------
    mini_batch : int, default: 1
        Mini batch size. Expecting the value in range [1,100] or so.

    loss : str,
        {'hinge', 'log', 'squared_hinge', 'modified_huber'} or regression loss
        {'squared_loss', 'quantile', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'},
        default: 'hinge'
        Loss function.

    max_iter : int, default: 100
        The maximum number of iterations. ``iter`` in Hivemall.

    disable_cv : bool, default: False
        Whether to disable convergence check.

    cv_rate : float, default: 0.005
        Threshold to determine convergence.

    optimizer : str, {'adagrad', 'sgd', 'adadelta', 'adam'},
        default: 'adagrad'
        Optimizer to update weights.

    eps : float, default: 1e-6
        Denominator value of AdaDelta/AdaGrad.

    rho : float, default: 0.95
        Decay rate of AdaDelta.

    regularization : str, {'rda', 'l1', 'l2', 'elasticnet'},
        default: 'rda'
        Regularization type.

    l1_ratio : float, default: 0.5
        Ratio of L1 regularizer as a part of Elastic Net regularization.

    alpha : float, default: 0.0001
        Regularization term. ``lambda`` in Hivemall.

    eta : str, {'inverse', 'fixed', 'simple'}, default: 'inverse'
        Learning rate scheme.

    eta0 : float, default: 0.1
        Initial learning rate.

    total_steps : int
        A total number of steps calculated by n_samples * n_epochs for simply
        scaled learning rate.

    power_t : float, default: 0.1
        Exponent for inversely scaled learning rate.

    scale : float, default: 100.0
        Scaling factor for cumulative weights.
    """

    loss_functions = {
        'hinge': (Hinge, 1.0),
        'log': (Log, ),
        'squared_hinge': (SquaredHinge, 1.0),
        'modified_huber': (ModifiedHuber, ),
    }
    loss_functions.update(GeneralRegressor.loss_functions)

    def __init__(self, loss='hinge', **kwargs):
        super(GeneralClassifier, self).__init__(**kwargs)

        if loss not in self.loss_functions:
            raise ValueError('The loss `%s` is not supported' % loss)
        loss_class, args = self.loss_functions[loss]
        self.loss_function = loss_class(*args)

    def train(self, x, y):
        pred = self.predict(x)[0]
        self.update(x, 1.0 if y > 0.0 else -1.0, pred)
