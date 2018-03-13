class RegularizationOptions(object):

    @staticmethod
    def setup(parser):
        parser.add_argument('-reg', '-regularization', dest='penalty',
                            choices=['rda', 'l1', 'l2', 'elasticnet'],
                            default='elasticnet',
                            help='regularization type (default: elasticnet)')
        parser.add_argument('-l1_ratio',
                            type=float, default=0.5,
                            help='ratio of L1 regularizer as a part of Elastic Net regularization (default: 0.5)')
        parser.add_argument('-lambda', dest='alpha',
                            type=float, default=0.0001,
                            help='regularization term (default: 0.0001)')
        return parser
