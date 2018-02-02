#coding=utf-8

import tensorflow as tf

def CNN(input_x, sequence_len, embedding_size, filter_sizes, num_filters):
    pooled_outputs = []
    input_x = tf.expand_dims(input_x, -1)
    for idx, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s"% (filter_size)):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            filter_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weight")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="filter_bias")
            
            # convolution (batch_size, sequence_len - filter_size + 1, in_channels, out_channnels)
            conv = tf.nn.conv2d(input_x, filter_weight, strides=[1,1,1,1], padding="VALID")

            # apply nonlinearity
            relu_output = tf.nn.relu(tf.nn.bias_add(conv, filter_bias))

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                relu_output,
                ksize=[1, sequence_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
            pooled_outputs.append(pooled)
    cnn_output = tf.squeeze(tf.concat(3, pooled_outputs), [1, 2])
    return cnn_output
