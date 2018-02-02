import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from bn_lstm import BNLSTMCell


try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

   
def bilstm(cell, x, scope=None):
    with tf.variable_scope(scope or "birnn") as scope:
        # input transformation
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unpack(input_x)

        output, _, _ = rnn.bidirectional_rnn(cell, cell, input_x, dtype=tf.float32)

        # output transformation to the original tensor type
        output = tf.pack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

def dynamic_rnn(cell, x, seq_len, scope=None):
    with tf.variable_scope(scope or "dynamic_rnn") as scope:
        # input transformation

        # define the forward and backward lstm cells
        outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)

        # output transformation to the original tensor type
        return outputs

def bidirectional_dynamic_rnn(cell, inputs_embedded, input_lengths, scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "dynamic_birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                            cell_bw=cell,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            scope=scope))
        outputs = tf.concat(2, (fw_outputs, bw_outputs))

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    1, (fw_state.c, bw_state.c), name='bidirectional_concat_c')
                state_h = tf.concat(
                    1, (fw_state.h, bw_state.h), name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat(1, (fw_state, bw_state),
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                    isinstance(bw_state, tuple) and
                    len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))


        state = concatenate_state(fw_state, bw_state)
        return outputs, state

