#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.layers as layers


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_loss(cnn, input_x, input_y):
    logits = cnn.inference(input_x)
    total_loss = cnn.loss(logits, input_y)
    accuracy = cnn.accuracy(logits, input_y)
    return logits, total_loss, accuracy

def cal_predictions(cnn, input_x):
    logits = cnn.inference(input_x)
    predictions = tf.argmax(logits, 1, name="predictions")
    return predictions


def attention(inputs, attention_size, l2_reg_lambda ):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):
        inputs = tf.concat(2, inputs)

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    #if l2_reg_lambda > 0:
    #    l2_loss += tf.nn.l2_loss(W_omega)
    #    l2_loss += tf.nn.l2_loss(b_omega)
    #    l2_loss += tf.nn.l2_loss(u_omega)
    #    tf.add_to_collection('losses', l2_loss)

    return output

def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape(
    )[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope) 
        vector_attn = tf.reduce_sum(tf.mul(input_projection, attention_context_vector), 2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.mul(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, 1)
        #outputs = max_pooling(weighted_projection, 4)

        return outputs


def max_pooling(lstm_out, height):
    #height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
    width = int(lstm_out.get_shape()[2])       # (step, length of input for one step)
    
    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
    	lstm_out,
    	ksize=[1, height, 1, 1],
    	strides=[1, 1, 1, 1],
    	padding='VALID')
    
    output = tf.reshape(output, [-1, width])
    
    return output
