import numpy as np


class EtaEstimator(object):

    def __init__(self, eta0):
        self.eta0 = eta0

    def eta(self, t):
        pass


class FixedEtaEstimatoer(EtaEstimator):

    def __init__(self, eta0):
        super(FixedEtaEstimatoer, self).__init__(eta0)

    def eta(self, t):
        return self.eta0


class SimpleEtaEstimator(EtaEstimator):

    def __init__(self, eta0, total_steps):
        super(SimpleEtaEstimator, self).__init__(eta0)
        self.min_eta = self.eta0 / 2.
        self.total_steps = total_steps

    def eta(self, t):
        if t > self.total_steps:
            return self.min_eta
        return self.eta0 / (1. + (t / self.total_steps))


class InverseScalingEtaEstimator(EtaEstimator):

    def __init__(self, eta0, power_t):
        super(InverseScalingEtaEstimator, self).__init__(eta0)
        self.power_t = power_t

    def eta(self, t):
        return self.eta0 / np.power(t, self.power_t)
