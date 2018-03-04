from .regularization import L1
from .regularization import L2
from .regularization import ElasticNet
from .eta_estimator import FixedEtaEstimatoer
from .eta_estimator import SimpleEtaEstimator
from .eta_estimator import InverseScalingEtaEstimator


class Optimizer(object):

    def __init__(self, regularization='rda', l1_ratio=0.5, alpha=0.0001,
                 eta='inverse', eta0=0.1, total_steps=None, power_t=0.1):
        regularization = regularization.lower()
        if regularization == 'l1':
            self.reg = L1(alpha=alpha)
        elif regularization == 'l2':
            self.reg = L2(alpha=alpha)
        elif regularization == 'elasticnet':
            self.reg = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
        elif regularization == 'rda':
            raise NotImplementedError("regularization 'rda' is not implemented yet")
        else:
            raise ValueError("regularization must be {'rda', 'l1', 'l2', 'elasticnet'}")

        eta = eta.lower()
        if eta == 'fixed':
            self.eta = FixedEtaEstimatoer(eta0=eta0)
        elif eta == 'simple':
            self.eta = SimpleEtaEstimator(eta0=eta0, total_steps=total_steps)
        elif eta == 'inverse':
            self.eta = InverseScalingEtaEstimator(eta0=eta0, power_t=power_t)
        else:
            raise ValueError("eta must be one of {'inverse', 'fixed', 'simple'}")
