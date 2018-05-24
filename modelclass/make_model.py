import tensorflow as tf
import utils


def make_model(dense_t, sparse_t, seq_len, params, mode):
    with tf.variable_scope("Normalization"):
        next_X_t = tf.layers.batch_normalization(
            dense_t, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope("RNN"):
        layers = [params["num_units"]] * params["num_layers"]
        if len(layers):
            if params["fw_lstm"]:
                fw_cells = utils.make_layers(layers=layers,
                                             dropout=params["dropout"],
                                             name="fw")

            if params["bw_lstm"]:
                bw_cells = utils.make_layers(layers=layers,
                                             dropout=params["dropout"],
                                             name="bw")

            if params["fw_lstm"] and params["bw_lstm"]:
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cells,
                                                                         cell_bw=bw_cells,
                                                                         inputs=next_X_t,
                                                                         sequence_length=seq_len[
                                                                             :, 0],
                                                                         dtype='float32')
                fw_state, bw_state = output_states
                _ = tf.concat(
                    [fw_state[-1][-1], bw_state[-1][-1]], -1)
                next_X_t = tf.concat(outputs, 2)

            elif params["fw_lstm"] or params["bw_lstm"]:
                if params["fw_lstm"]:
                    cells = fw_cells
                else:
                    cells = bw_cells
                (next_X_t,  output_state) = tf.nn.dynamic_rnn(cell=cells, inputs=next_X_t,
                                                              sequence_length=seq_len[:, 0],
                                                              dtype='float32')

    with tf.variable_scope("fc_layer_entity") as scope:
        deep_and_wide = tf.concat([next_X_t, sparse_t], axis=2)
        w = tf.Variable(utils.xavier_init(
            (deep_and_wide.shape[-1].value, params["num_entities"])), name='weights')
        b = tf.Variable(tf.zeros(params["num_entities"]), name='biases')
        scope.reuse_variables()

        deep_and_wide_unstack = tf.unstack(deep_and_wide, axis=1)
        fc_out = []

        for time_step in range(deep_and_wide.shape[1]):
            # applying the same fc layer to every time step
            fc_out.append(tf.nn.relu(
                tf.matmul(deep_and_wide_unstack[time_step], w) + b))
        logits = tf.stack(fc_out, axis=1)

    return logits
