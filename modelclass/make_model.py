import tensorflow as tf
from . import utils  # pylint : ignore


def make_model(deep_t, wide_t, params, mode):

    X = utils.batch_norm(deep_t, mode=mode)

    X = utils.conv_2d(X, 32, "1")
    # X = tf.layers.dropout(
    #     inputs=X, rate=params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
    X = utils.conv_2d(X, 64, "2")
    X = tf.reshape(X, [-1, 7 * 7 * 64])

    with tf.variable_scope("fc_1"):
        X = tf.layers.dense(inputs=X, units=1024, activation=tf.nn.relu)
        X = tf.layers.dropout(
            inputs=X, rate=params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope("fc_2"):
        deep_wide = tf.concat([wide_t, X], 1)
        logits = tf.layers.dense(
            inputs=deep_wide, units=10, activation=tf.nn.relu)

    return logits
