import numpy as np
import metrics
import tensorflow as tf
import utils
from modelclass.make_model import make_model


def model_fn(features, labels, mode, params):
    """Builds the graph, the train operation, and the metric operations

    Args:
      features: A dict of feature_column w/:
            * "Dense": (batch_size, max_words, dense_size), np.float32
            * "Sparse": (batch_size, max_words, sparse_size), np.float32
            * "seq_len": (batch_size, 1), np.int32
      labels: (batch_size, max_words), np.int32
      mode: tf.estimator.ModeKeys.[TRAIN, EVAL, PREDICT]
      params: a Dictionary of configuration parameters

    Returns:
      tf.estimator.EstimatorSpec
    """

    dense_t = tf.feature_column.input_layer(
        features, params['feature_columns'][0])
    dense_t = tf.reshape(
        dense_t, shape=(-1, params["max_words"], params["dense_size"]))

    sparse_t = tf.feature_column.input_layer(
        features, params['feature_columns'][1])
    sparse_t = tf.reshape(
        sparse_t, shape=(-1, params["max_words"], params["sparse_size"]))

    seq_len = tf.feature_column.input_layer(
        features, params['feature_columns'][2])
    seq_len = tf.cast(seq_len, tf.int32)
    # label = tf.feature_column.input_layer(        features, params['feature_columns'][2])
    # tf.feature_column.input_layer(
    #     features, params['feature_columns'][2])
    logits = make_model(dense_t, sparse_t, seq_len, params, mode)

    predicted_classes = tf.argmax(logits, -1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    weight_vector = tf.sequence_mask(
        seq_len[:, 0], dtype=tf.float32, maxlen=params["max_words"])
    # label_t = tf.feature_column.input_layer(
    #     labels, params['feature_columns'][3])
    # # label_t = labels
    label_t = tf.cast(labels, tf.int32)
    label_t = tf.reshape(label_t, (-1, params["max_words"]))
    print("label_t", label_t)
    loss = tf.contrib.seq2seq.sequence_loss(logits=logits,  # pylint:ignore
                                            targets=label_t,
                                            weights=weight_vector,
                                            name="calc_loss"
                                            )

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=label_t,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metric_ops = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    metric_ops.update(metrics.f_score_tf(label_t, predicted_classes))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metric_ops)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    train_op = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]).minimize(
        loss, global_step=tf.train.get_global_step())

    print("Trainable variables", np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
