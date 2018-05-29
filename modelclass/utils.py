import tensorflow as tf


def conv_2d(X, channels, name=""):
    # makes a convolutional block inlc. a pooling layer.
    with tf.variable_scope("Conv_" + name):
        # Convolutional Layer #1
        conv = tf.layers.conv2d(
            inputs=X,
            filters=channels,
            kernel_size=[7, 7],
            padding="same",
            activation=tf.nn.relu)
        # RELU Layer
        # relu = tf.layers.dense(
        #     inputs=conv, units=channels, activation=tf.nn.relu)
        # Pooling Layer #1
        pool = tf.layers.max_pooling2d(
            inputs=conv, pool_size=[2, 2], strides=2)

    return pool


def batch_norm(X, mode, name=""):
    with tf.variable_scope("Normalization_" + name):
        return tf.layers.batch_normalization(
            X, training=mode == tf.estimator.ModeKeys.TRAIN)
