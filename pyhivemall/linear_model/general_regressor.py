from sklearn.base import RegressorMixin
from sklearn.linear_model.sgd_fast import SquaredLoss
from sklearn.linear_model.sgd_fast import EpsilonInsensitive
from sklearn.linear_model.sgd_fast import SquaredEpsilonInsensitive
from sklearn.linear_model.sgd_fast import Huber

from .general_learner_base import GeneralLearnerBase
from .sgd_fast import Quantile


class GeneralRegressor(GeneralLearnerBase, RegressorMixin):

    """Regression using generic optimizer

    Parameters
    ----------
    mini_batch : int, default: 1
        Mini batch size. Expecting the value in range [1,100] or so.

    loss : str,
        {'squared_loss', 'quantile', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'},
        default: 'squared'
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
        'squared_loss': (SquaredLoss, ),
        'quantile': (Quantile, 0.5),
        'epsilon_insensitive': (EpsilonInsensitive, 0.1),
        'squared_epsilon_insensitive': (SquaredEpsilonInsensitive, 0.1),
        'huber': (Huber, 1.0),
    }

    def __init__(self, loss='squared_loss', **kwargs):
        super(GeneralRegressor, self).__init__(**kwargs)

        if loss not in self.loss_functions:
            raise ValueError('The loss `%s` is not supported' % loss)
        loss_class, args = self.loss_functions[loss]
        self.loss_function = loss_class(*args)

    def train(self, x, y):
        pred = self.predict(x)[0]
        self.update(x, y, pred)
