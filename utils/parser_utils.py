class ParserClass(object):
    def __init__(self, parser):
        """
        Parses arguments and saves them in the Parser Class
        :param parser: A parser to get input from
        """
        parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='batch_size for experiment')
        parser.add_argument('--epochs', type=int, nargs="?", default=100, help='Number of epochs to train for')
        parser.add_argument('--logs_path', type=str, nargs="?", default="classification_logs/",
                            help='Experiment log path, '
                                 'where tensorboard is saved, '
                                 'along with .csv of results')
        parser.add_argument('--experiment_prefix', nargs="?", type=str, default="classification",
                            help='Experiment name without hp details')
        parser.add_argument('--continue_epoch', nargs="?", type=int, default=-1, help="ID of epoch to continue from, "
                                                                                      "-1 means from scratch")
        parser.add_argument('--tensorboard_use', nargs="?", type=str, default="False",
                            help='Whether to use tensorboard')
        parser.add_argument('--dropout_rate', nargs="?", type=float, default=0.5, help="Dropout value")
        parser.add_argument('--embedding_dim', nargs="?", type=int, default=300, help="Embedding dimension")
        parser.add_argument('--filter_sizes', nargs="?", type=list, default=[3,4,5], help='Filter sizes')
        parser.add_argument('--num_filters', nargs="?", type=int, default=100, help="Number of feature maps per filter")
        parser.add_argument('--pt_embeddings', nargs="?", type=str, default="False", help='Whether to use pretrained embeddings')
        parser.add_argument('--seed', nargs="?", type=int, default=1122017, help='Whether to use tensorboard')

        self.args = parser.parse_args()

    def get_argument_variables(self):
        """
        Processes the parsed arguments and produces variables of specific types needed for the experiments
        :return: Arguments needed for experiments
        """
        batch_size = self.args.batch_size
        experiment_prefix = self.args.experiment_prefix
        seed = self.args.seed
        dropout_rate = self.args.dropout_rate
        tensorboard_enable = True if self.args.tensorboard_use == "True" else False
        continue_from_epoch = self.args.continue_epoch  # use -1 to start from scratch
        epochs = self.args.epochs
        logs_path = self.args.logs_path
        pt_embeddings=False if self.args.pt_embeddings == 'False' else True
        embedding_dim = self.args.embedding_dim
        filter_sizes = self.args.filter_sizes
        num_filters = self.args.num_filters
        
        return batch_size, seed, epochs, logs_path, continue_from_epoch, tensorboard_enable, embedding_dim, \
            filter_sizes, num_filters, experiment_prefix, dropout_rate, pt_embeddings
