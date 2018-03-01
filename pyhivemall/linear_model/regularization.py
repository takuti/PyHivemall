class Regularization(object):

    def __init__(self, alpha=0.0001):
        self.alpha = alpha

    def regularize(self, weight, gradient):
        return gradient + self.alpha * self.get_regularizer(weight)

    def get_regularizer(self, weight):
        pass


class L1(Regularization):

    def get_regularizer(self, weight):
        return 1. if weight > 0. else -1.


class L2(Regularization):

    def get_regularizer(self, weight):
        return weight


class ElasticNet(Regularization):

    def __init__(self, l1_ratio=0.5, alpha=0.0001):
        super(ElasticNet, self).__init__(alpha=alpha)
        self.l1 = L1
        self.l2 = L2

        assert 0. <= l1_ratio and l1_ratio <= 1., 'L1 ratio should be in [0.0, 1.0], got: ' + str(l1_ratio)
        self.l1_ratio = l1_ratio

    def get_regularizer(self, weight):
        return self.l1_ratio * self.l1.get_regularizer(weight) + (1. - self.l1_ratio) * self.l2.get_regularizer(weight)
