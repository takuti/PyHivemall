class OptimizerOptions(object):

    @staticmethod
    def setup(parser):
        parser.add_argument('-iter', '-iterations', dest='max_iter',
                            type=int, default=100,
                            help='the maximum number of iterations (default: 100)')
        # parser.add_argument('-disable_cv', '-disable_cvtest',
        #                     type=bool, default=False,
        #                     help='whether to disable convergence check [unsupported]')
        # parser.add_argument('-cv_rate', '-convergence_rate',
        #                     type=float, default=0.005,
        #                     help='threshold to determine convergence [unsupported]')
        # parser.add_argument('-opt', '-optimizer',
        #                     choices=['adagrad', 'sgd', 'adadelta', 'adam'],
        #                     default='sgd',
        #                     help='optimizer (default: sgd)')
        # parser.add_argument('-eps',
        #                     type=float, default=1e-6,
        #                     help='denominator value of AdaDelta/AdaGrad (default: 1e-6)')
        # parser.add_argument('-rho',
        #                     type=float, default=0.95,
        #                     help='decay rate of AdaDelta (default: 0.95)')
        return parser
