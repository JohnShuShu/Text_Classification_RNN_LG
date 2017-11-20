import tensorflow as tf

""""
author: Wen Cui
Date: Nov 11, 2017
Experiemnts: last_rnn_output, bidirectional lstm, initial_state(learning fast), dropout_keep_prob(better), SGD optimizer(worse)
"""
import tensorflow as tf
import numpy as np

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def loadGlove(model_file):
    # Get the index dict
    embeddings_index = []
    vocab = []
    f = open(model_file)
    for line in f:
        row = line.split(' ')
        vocab.append(row[0])
        embeddings_index.append(np.array(row[1:], dtype='float32'))
    f.close()
    embedding_index = np.vstack(embeddings_index)
    return vocab, embedding_index


class TextRNN:
    def __init__(self, num_classes, seq_len, learning_rate, decay_steps, decay_rate, embedding_size, voc_size, pretrain_bool, pretrained_embedding, bidirectional=False):
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.embedding_size = embedding_size
        self.voc_size = voc_size
        self.bidirectional = bidirectional
        self.pretrain = pretrain_bool
        self.pretrained_embedding = pretrained_embedding
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        # add placeholder (X,label)
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x") # [batch_size, seq_len, embed_size]
            self.y = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y") #[batch_size, num_classes]
        self.batch_size = tf.shape(self.x)[0]
        # set init_state as variable(has to be tuple) so that can learn it while training
        init_state = tf.get_variable('init_state', [1, self.embedding_size], initializer=self.initializer)
        init_state = tf.tile(init_state, [self.batch_size, 1])
        self.init_state = (init_state, init_state)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        self.logits = self.adding_layers()
        with tf.name_scope("softmax"):
            self.predictions = tf.nn.softmax(self.logits)

        with tf.name_scope('Accuracy'):
            correct = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="Accuracy")

        # Simply cross entropy loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        # Define learning_rate and optimizer
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True, name='learning_rate')

        with tf.name_scope('train'):
            self.train_step = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam")


    def adding_layers(self):
        """Structure graph here: 1. self-learned embeddding layer/ pretrained,
                                2.LSTM/ Bi-LSTM layer 3.ff output 4.softmax 
         """
        # 1. Embedding layer
        # TODO: pretrained model
        if self.pretrain:
            W = tf.Variable(tf.constant(0.0, shape=[self.voc_size, self.embedding_size]), trainable=False, name="pretrained_embedding")
            # self.embedding_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.voc_size, self.embedding_size], name='embedding_placeholder')
            self.Embedding = W.assign(self.pretrained_embedding)
            self.rnn_inputs = tf.nn.embedding_lookup(self.Embedding, self.x)
        else:
            self.Embedding = tf.get_variable("Embedding", shape=[self.voc_size, self.embedding_size], initializer=self.initializer)
            self.rnn_inputs = tf.nn.embedding_lookup(self.Embedding, self.x, name='rnn_inputs') #shape:[batch_size,seq_len,embed_size]

        # 2. bi-LSTM layer
        if self.bidirectional:
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size)  # forward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.rnn_inputs, dtype=tf.float32)
            # concat output
            rnn_outputs = tf.concat(outputs, axis=2, name='bidirectional_rnn_outputs')  # [batch_size,sequence_length,hidden_size*2]
            # self.last_rnn_output = rnn_outputs[:, -1, :]
            self.last_rnn_output = tf.reduce_mean(rnn_outputs, axis=1, name='last_output')
        # 2. LSTM layer
        else:
            self.rnn_inputs = tf.unstack(self.rnn_inputs, self.seq_len, 1, name='rnn_inputs')# list of batch_size*embed_size tensor, len(list) = seq_len

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            rnn_outputs, final_state = tf.contrib.rnn.static_rnn(lstm_cell, self.rnn_inputs, initial_state=self.init_state, dtype=tf.float32)

            # Add dropout
            rnn_outputs = tf.nn.dropout(rnn_outputs, self.dropout_keep_prob, name='rnn_dropout')
            # self.last_rnn_output = rnn_outputs[-1,:,:] #shape: batch_size * embed_size
            self.last_rnn_output = tf.reduce_mean(rnn_outputs, axis=0, name='rnn_last_output')


        # 3. Output layer
        with tf.variable_scope('output'):
            if self.bidirectional:
                W = tf.get_variable('W', [self.embedding_size*2, self.num_classes], initializer=self.initializer)
            else:
                W = tf.get_variable('W', [self.embedding_size, self.num_classes], initializer=self.initializer)
            b = tf.get_variable('b', [self.num_classes], initializer=self.initializer)
        logits = tf.matmul(self.last_rnn_output, W) + b
        return logits


