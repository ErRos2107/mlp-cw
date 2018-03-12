import tensorflow as tf                        

from network_architectures import VGGClassifier
from network_architectures import TextClassifier
from network_architectures import RNNClassifier

class ClassifierNetworkGraph:
    def __init__(self, input_x, target_placeholder, dropout_rate, embeddings, vocab_size, max_sent_length, cell,
                 hidden_unit, typec, num_units,
                 embedding_dim=300, num_filters=100, filter_sizes=[3,4,5],
                 batch_size=100, num_channels=1, n_classes=14, l2_norm=3, activation='relu', is_training=True,
                 tensorboard_use=False, l2_reg_lambda=0.0, use_rnn=True):

        """
        Initializes a Classifier Network Graph that can build models, train, compute losses and save summary statistics and images
        :param input_x: A placeholder that will feed the input text, usually of size [batch_size, padded_length]
        :param target_placeholder: A target placeholder of size [batch_size,]. The classes should be in index form
               i.e. not one hot encoding, that will be done automatically by tf
        :param dropout_rate: A placeholder of size [None] that holds a single float that defines the amount of dropout
               to apply to the network. i.e. for 0.1 drop 0.1 of neurons
        :param batch_size: The batch size
        :param num_channels: Number of channels
        :param n_classes: Number of classes we will be classifying
        :param is_training: A placeholder that will indicate whether we are training or not
        :param tensorboard_use: Whether to use tensorboard in this experiment
        :param use_batch_normalization: Whether to use batch normalization between layers
        :param strided_dim_reduction: Whether to use strided dim reduction instead of max pooling
        """
        self.batch_size = batch_size       
        if typec=='cnn':
            self.c = TextClassifier(self.batch_size, filter_sizes=filter_sizes, name="classifier_neural_network", embeddings=embeddings, max_sent_length=max_sent_length,
                               vocab_size=vocab_size, num_channels=num_channels, embedding_dim=embedding_dim, num_filters=num_filters,
                               num_classes=n_classes, l2_norm=l2_norm, activation=activation, num_units=num_units)
        elif typec=='rnn':
            self.c=RNNClassifier(self.batch_size, name='classifier_rnn', embeddings=embeddings, max_sent_length=max_sent_length,
                                 vocab_size=vocab_size, embedding_dim=embedding_dim, num_classes=n_classes, cell=cell,
                                 num_units=num_units, hidden_unit=hidden_unit)

        self.input_x = input_x
        self.dropout_rate = dropout_rate
        self.targets = target_placeholder

        self.training_phase = is_training
        self.n_classes = n_classes
        self.iterations_trained = 0

        self.is_tensorboard = tensorboard_use
        self.l2_reg_lambda = l2_reg_lambda
        
    def loss(self):
        """build models, calculates losses, saves summary statistcs and images.
        Returns:
            dict of losses.
        """
        with tf.name_scope("losses"):
            text_inputs = self.input_x  # conditionally apply augmentaions
            true_outputs = self.targets
            # produce predictions and get layer features to save for visual inspection
            preds, layer_features = self.c(text_input=text_inputs, training=self.training_phase,
                                           dropout_rate=self.dropout_rate)
            # compute loss and accuracy
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(true_outputs, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            crossentropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=true_outputs))

            # add loss and accuracy to collections
            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

            # save summaries for the losses, accuracy and image summaries for input images, augmented images
            # and the layer features
            #self.save_features(name="VGG_features", features=layer_features)
            #tf.summary.image('text', [tf.concat(tf.unstack(self.input_x, axis=0), axis=0)])
            #tf.summary.image('augmented_image', [tf.concat(tf.unstack(image_inputs, axis=0), axis=0)])
            tf.summary.scalar('crossentropy_losses', crossentropy_loss)
            tf.summary.scalar('accuracy', accuracy)

        return {"crossentropy_losses": tf.add_n(tf.get_collection('crossentropy_losses'),
                                                name='total_classification_loss'),
                "accuracy": tf.add_n(tf.get_collection('accuracy'), name='total_accuracy')}

    def save_features(self, name, features, num_rows_in_grid=4):
        """
        Saves layer features in a grid to be used in tensorboard
        :param name: Features name
        :param features: A list of feature tensors
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = num_rows_in_grid
            x_channels = int(channels / y_channels)

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                        y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            tf.summary.image('{}_{}'.format(name, i), activations_features)


    def train(self, losses, learning_rate=1e-3, beta1=0.9):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
        grads_and_vars = c_opt.compute_gradients(losses["crossentropy_losses"])
        c_error_opt_op = c_opt.apply_gradients(grads_and_vars)
        
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        #with tf.control_dependencies(update_ops):
         #   c_error_opt_op = c_opt.minimize(losses["crossentropy_losses"], var_list=self.c.variables,
          #                                  colocate_gradients_with_ops=True)

        return c_error_opt_op

    def init_train(self):
        """
        Builds graph ops and returns them
        :return: Summary, losses and training ops
        """
        losses_ops = self.loss()
        c_error_opt_op = self.train(losses_ops)
        summary_op = tf.summary.merge_all()
        return summary_op, losses_ops, c_error_opt_op
