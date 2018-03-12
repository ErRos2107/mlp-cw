import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib import rnn
from tensorflow.python.ops.nn_ops import leaky_relu

from utils.network_summary import count_parameters


class VGGClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=2, strided_dim_reduction=True):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False

    def __call__(self, text_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param text_input: Text input to produce embeddings for. e.g. for text data [batch_size, 300]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('VGGNet'):
                outputs = image_input
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if (j == self.inner_layer_depth-1) and self.strided_dim_reduction:
                                    stride = 2
                                else:
                                    stride = 1
                                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3],
                                                           strides=(stride, stride),
                                                           padding='SAME', activation=None)
                                outputs = leaky_relu(outputs, name="leaky_relu{}".format(i))
                                layer_features.append(outputs)
                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                        if self.strided_dim_reduction==False:
                            outputs = tf.layers.max_pooling2d(outputs, pool_size=(2, 2), strides=2)

                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                                                                              # apply dropout only at dimensionality
                                                                              # reducing steps, i.e. the last layer in
                                                                              # every group

            c_conv_encoder = outputs
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "VGGNet")

        return c_conv_encoder, layer_features


class TextClassifier:
    def __init__(self, batch_size, filter_sizes, name, num_classes, embeddings, max_sent_length, vocab_size,num_units, num_channels=1,
                 embedding_dim=300, num_filters=100, l2_norm=3, activation='relu'):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param filter_sizes: A list containing the filters sizes for the convolutional layer
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param embeddings: the pretrained embeddings
        :param embed_size: the size of each embeddings
        :param num_filters: number of filter per size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.filter_sizes = filter_sizes
        self.name = name
        self.num_classes = num_classes
        self.build_completed = False
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.max_sent_length = max_sent_length
        self.vocab_size = vocab_size
        self.l2_norm = l2_norm
        self.activation=activation
        self.num_units = num_units
            
    def __call__(self, text_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param text_input: Text input to produce embeddings for. e.g. for text data [batch_size, sequence_length]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('VGGNet'):
                if self.embeddings==None:
                     with tf.device('/cpu:0'), tf.name_scope("embedding"):
                        W = tf.Variable(
                            tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                            name="W")             
                else:
                    W = self.embeddings

                embedded_chars = tf.nn.embedding_lookup(W, text_input)
                inputs = embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                pooled_outputs = []
                for i, filter_size in enumerate(self.filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                        conv = tf.nn.conv2d(
                            inputs,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        if self.activation=='relu':
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        elif self.activation=='sigmoid':
                            h = tf.sigmoid(tf.nn.bias_add(conv, b), name="sigmoid")
                        elif self.activation=='tanh':
                            h = tf.tanh(tf.nn.bias_add(conv, b), name="tanh")
                        layer_features.append(h)
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, self.max_sent_length - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = self.num_filters * len(self.filter_sizes)
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                if self.l2_norm != 0:
                    h_pool_flat = self.l2_norm * tf.divide(h_pool_flat, tf.norm(h_pool_flat, ord='euclidean'))
                # Add dropout
                with tf.name_scope("dropout"):
                    h_drop = tf.layers.dropout(h_pool_flat, rate=dropout_rate, training=training)

            c_conv_encoder = h_drop
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)

            dense = tf.layers.dense(inputs=c_conv_encoder, units=self.num_units, activation=tf.nn.relu)

            # Logits Layer
            scores = tf.layers.dense(inputs=dense, units=self.num_classes)

        
        #self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "VGGNet")

        return scores, layer_features



class RNNClassifier:
    def __init__(self, batch_size, name, num_classes, embeddings, max_sent_length, vocab_size, cell, num_units,hidden_unit=100,
                 embedding_dim=300):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param filter_sizes: A list containing the filters sizes for the convolutional layer
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param embeddings: the pretrained embeddings
        :param embed_size: the size of each embeddings
        :param num_filters: number of filter per size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.name = name
        self.num_classes = num_classes
        self.build_completed = False
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_sent_length = max_sent_length
        self.vocab_size = vocab_size
        self.hidden_unit = hidden_unit
        self.cell = cell
        self.num_units= num_units

    def __call__(self, text_input, training=False, dropout_rate=0.0):

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('VGGNet'):
                if self.embeddings==None:
                     with tf.name_scope("embedding"):
                        W = tf.Variable(
                            tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                            name="W")             
                else:
                    W = self.embeddings

                embedded_chars = tf.nn.embedding_lookup(W, text_input)
                if self.cell=='bidlstm':
                    lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_unit) #forward direction cell
                    lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_unit) #backward direction cell
                    if dropout_rate is not None:
                        lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=1-dropout_rate)
                        lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=1-dropout_rate)
                    outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,embedded_chars,dtype=tf.float32) 
                    output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
                    output_rnn_last=tf.reduce_mean(output_rnn,axis=1) #[batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
                elif self.cell=='lstm':
                    lstm_cell=rnn.BasicLSTMCell(self.hidden_unit)
                    if dropout_rate is not None:
                        lstm_cell=rnn.DropoutWrapper(lstm_cell,output_keep_prob=1-dropout_rate)
                    outputs,_ = tf.nn.dynamic_rnn(lstm_cell, embedded_chars, dtype=tf.float32)
                    output_rnn=tf.concat(outputs,axis=2)
                    output_rnn_last=tf.reduce_mean(output_rnn,axis=1)
                elif self.cell=='gru':
                    gru_cell=rnn.GRUCell(self.hidden_unit)
                    if dropout_rate is not None:
                        gru_cell=rnn.DropoutWrapper(gru_cell,output_keep_prob=1-dropout_rate)
                    outputs,_ = tf.nn.dynamic_rnn(gru_cell, embedded_chars, dtype=tf.float32)
                    output_rnn=tf.concat(outputs,axis=2)
                    output_rnn_last=tf.reduce_mean(output_rnn,axis=1)

            c_conv_encoder = output_rnn_last
            dense = tf.layers.dense(inputs=c_conv_encoder, units=self.num_units, activation=tf.nn.relu)

            # Logits Layer
            scores = tf.layers.dense(inputs=dense, units=self.num_classes, activation=tf.sigmoid)
        
        #self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "VGGNet")

        return scores, layer_features
