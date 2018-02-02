import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from utils import attention, task_specific_attention, max_pooling
from LSTM import bilstm, dynamic_rnn, bidirectional_dynamic_rnn
from cnn import CNN

class Polymerization(object):
    def __init__(self, cell, embeddings, embedding_size, attention_dim, rnn_size, num_rnn_layers, num_classes, num_unroll_steps, class_weight, max_grad_norm, lr, num_filters, filter_sizes, l2_reg_lambda=1e-4):
        # define input variable
        self.cell = cell
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.attention_dim = attention_dim
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.l2_reg_lambda = l2_reg_lambda
        self.num_unroll_steps = num_unroll_steps
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        self.input_data = tf.placeholder(tf.int32,[None, None, None], name="inputs")#[batch_size, sent, word]
        self.target = tf.placeholder(tf.int64,[None], name="labels")#label
        self.sent_lengths = tf.placeholder(tf.int32,[None, None], name="sent_lengths")#(session_len, sent_len)
        self.session_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sess_lengths')#session_len
        self.dropout_ratio = tf.placeholder(tf.float32, name='dropout')#session_len

    def inference(self, keep_prob, is_training):
        #self.sent_size, self.num_unroll_steps represent the length after padding
        (self.batch_size, self.sent_size, _) = tf.unpack(tf.shape(self.input_data))
        word_level_lengths = tf.reshape(self.sent_lengths, [self.batch_size * self.sent_size])

        #embedding layer
        with tf.device("/cpu:0"),tf.variable_scope("embedding_layer"):
            embedded = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="embeddings")
            inputs_embedded = tf.nn.embedding_lookup(embedded, self.input_data)
        word_level_inputs = tf.reshape(inputs_embedded, [self.batch_size * self.sent_size, self.num_unroll_steps, self.embedding_size])

        # instance LSTM
        #lstm = LSTM(self.rnn_size, self.num_rnn_layers, keep_prob, is_training)

        with tf.variable_scope("word_layer") as scope:
            word_level_output = CNN(word_level_inputs, self.num_unroll_steps, self.embedding_size, self.filter_sizes, self.num_filters)
            #word_encoder_output, _ = bidirectional_dynamic_rnn(self.cell, word_level_inputs, word_level_lengths, scope)
            #word_encoder_output = bilstm(self.cell, word_level_inputs, scope)
            #(batch_size * self.sent_size, self.embedding_size)
            #with tf.variable_scope('attention') as att_scope:
            #    word_level_output = task_specific_attention(word_encoder_output, self.attention_dim, scope=att_scope)
            #
            with tf.variable_scope('dropout'):
                #word_level_output = layers.dropout(word_level_output, keep_prob=keep_prob, is_training=is_training)
                word_level_output = tf.nn.dropout(word_level_output, self.dropout_ratio)
            #sents = attention(word_dense, self.attention_dim, self.l2_reg_lambda)

        sentences_inputs = tf.reshape(word_level_output, [self.batch_size, self.sent_size, self.num_filters * len(self.filter_sizes)])
        with tf.variable_scope("sent_layer") as scope:
            sent_encoder_output = dynamic_rnn(self.cell, sentences_inputs, self.session_lengths, scope)

            #(batch_size * self.sent_size, self.embedding_size)
            #sentence_level_output = max_pooling(sent_encoder_output, 4)
            #sentence_level_output = sent_encoder_output[:,-1,:]
            with tf.variable_scope('attention') as att_scope:
                sentence_level_output = task_specific_attention(sent_encoder_output, self.attention_dim, scope=att_scope)
                
            with tf.variable_scope('dropout'):
                #sentence_level_output = layers.dropout(sentence_level_output, keep_prob=keep_prob, is_training=is_training)
                sentence_level_output = tf.nn.dropout(sentence_level_output, self.dropout_ratio)
           #sents = attention(word_dense, self.attention_dim, self.l2_reg_lambda)
           


        with tf.name_scope("linear_layer"):
            softmax_w = tf.get_variable("softmax_w", initializer=tf.truncated_normal([self.attention_dim, self.num_classes], stddev=0.1))
            softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(0., shape=[1]))
            logits = tf.matmul(sentence_level_output, softmax_w) + softmax_b
            l2_loss = 0
            if self.l2_reg_lambda>0:
                l2_loss += tf.nn.l2_loss(softmax_w)
                l2_loss += tf.nn.l2_loss(softmax_b)
                weight_decay = tf.mul(l2_loss, self.l2_reg_lambda, name='l2_loss')
                tf.add_to_collection('losses', weight_decay)
        return logits, l2_loss

    def loss_and_acc(self, logits, l2_loss):
        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.target)
            #labelids = tf.to_int32(self.target)
            #weights = tf.nn.embedding_lookup(self.class_weight, labelids)
            #cross_entropy = tf.mul(loss, weights)

            #self.cost = tf.reduce_mean(self.loss)
            cost = tf.reduce_mean(loss) + self.l2_reg_lambda * l2_loss


        with tf.name_scope("accuracy"):
            prediction = tf.argmax(logits,1)
            correct_prediction = tf.equal(prediction,self.target)
            correct_num =tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

        return cost, prediction, correct_num, accuracy

    def grad(self, cost, accuracy):
        #add summary
        loss_summary = tf.scalar_summary("loss", cost)
        #add summary
        accuracy_summary=tf.scalar_summary("accuracy_summary", accuracy)

        globle_step = tf.Variable(0,name="globle_step",trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      self.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        summary =tf.merge_summary([loss_summary,accuracy_summary,grad_summaries_merged])

        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        train_op=optimizer.apply_gradients(zip(grads, tvars))
       
        return train_op, summary

