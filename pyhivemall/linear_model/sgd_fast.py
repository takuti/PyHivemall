from sklearn.linear_model.sgd_fast import Regression


class Quantile(Regression):
    """Quantile loss.

    This is useful to predict rank/order and you do not mind the mean error to
    increase as long as you get the relative order correct.

    http://en.wikipedia.org/wiki/Quantile_regression
    """

    def __init__(self, tau):
        assert 0. < tau and tau < 1., 'tau must be in range (0, 1), but: ' + str(tau)
        self.tau = tau

    def loss(self, p, y):
        e = y - p
        if e > 0.:
            return self.tau * e
        else:
            return -(1. - self.tau) * e

    def _dloss(self, p, y):
        e = y - p
        if e == 0.:
            return 0.
        return -self.tau if e > 0. else (1. - self.tau)

    def __reduce__(self):
        return Quantile, (self.tau,)
