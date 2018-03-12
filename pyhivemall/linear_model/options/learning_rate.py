class LearningRateOptions(object):

    @staticmethod
    def setup(parser):
        parser.add_argument('-eta', dest='learning_rate',
                            choices=['inverse', 'fixed', 'simple'],
                            default='inverse',
                            help='learning rate schema (default: inverse)')
        parser.add_argument('-eta0',
                            type=float, default=0.1,
                            help='initial learning rate (default: 0.1)')
        # parser.add_argument('-t', '-total_steps',
        #                     type=int,
        #                     help='a total number of steps calculated by n_samples * n_epochs for simply scaled learning rate')
        parser.add_argument('-power_t',
                            type=float, default=0.1,
                            help='exponent for inversely scaled learning rate (default: 0.1)')
        # parser.add_argument('-scale',
        #                     type=float, default=100.0,
        #                     help='scaling factor for cumulative weights (default: 100.0)')
        return parser
