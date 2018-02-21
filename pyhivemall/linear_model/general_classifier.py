from sklearn.base import ClassifierMixin

from .general_learner_base import GeneralLearnerBase


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

    def __init__(self, mini_batch=1, loss='hinge', max_iter=100,
                 disable_cv=False, cv_rate=0.005, optimizer='adagrad', eps=1e-6,
                 rho=0.95, regularization='rda', l1_ratio=0.5, alpha=0.0001,
                 eta='inverse', eta0=0.1, total_steps=None, power_t=0.1,
                 scale=100.0):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
