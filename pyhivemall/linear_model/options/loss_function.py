class LossFunctionOptions(object):

    @staticmethod
    def setup(parser, classification=False):
        loss_functions = ['squared', 'quantile', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'huber']
        if classification:
            loss_functions += ['hinge', 'log', 'squared_hinge', 'modified_huber']
        parser.add_argument('-loss', '-loss_function',
                            choices=loss_functions,
                            default='squared',
                            help='loss function (default: squared)')
        return parser
