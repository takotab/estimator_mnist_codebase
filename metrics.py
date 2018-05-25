import tensorflow as tf
import utils
ENTITIES = [str(i) for i in range(10)]


def extra_metrics(labels, predictions):
    # https://stackoverflow.com/a/48452770/6950549
    if type(predictions) is dict:
        predictions = tf.cast(predictions["class_ids"], tf.int64)
    metric_ops = {}
    n = len(ENTITIES)
    recall = [0] * n

    precision = [0] * n

    _f_score = [0] * n
    f_score = [0] * n
    labels = tf.cast(labels, tf.int32)

    with tf.variable_scope("Metrics"):
        for k in range(n):

            recall[k] = tf.metrics.recall(
                labels=tf.equal(labels, k),
                predictions=tf.equal(predictions, k),
            )

            metric_ops['recall_' + ENTITIES[k]] = recall[k]
            tf.summary.scalar('recall_' + ENTITIES[k], recall[k][1])

            precision[k] = tf.metrics.precision(
                labels=tf.equal(labels, k),
                predictions=tf.equal(predictions, k)
            )
            metric_ops['precision_' + ENTITIES[k]] = precision[k]
            tf.summary.scalar(
                'precision_' + ENTITIES[k], precision[k][1])

            _f_score[k] = 2 * (precision[k][1] * recall[k][1] /
                               (precision[k][1] + recall[k][1]))
            _f_score[k] = tf.where(tf.is_nan(_f_score[k]),
                                   tf.zeros_like(_f_score[k]), _f_score[k])
            f_score[k] = tf.metrics.mean(
                _f_score[k])

            metric_ops['f_score_' + ENTITIES[k]] = f_score[k]
            tf.summary.scalar(
                'f_score_' + ENTITIES[k], f_score[k][1])

    f_score_stacked = tf.stack(f_score)
    f_score_total = tf.metrics.mean(f_score_stacked)

    accuracy = tf.metrics.accuracy(labels, predictions)

    with tf.variable_scope("Overal_Metrics"):
        tf.summary.scalar('f_score', f_score_total[1])
        tf.summary.scalar("Accuracy", accuracy[1])

    metric_ops['f_score_total'] = f_score_total
    metric_ops['Accuracy'] = accuracy

    # accuracy = tf.metrics.accuracy(labels=labels,
    #                                predictions=predictions,
    #                                name='acc_op')
    # metric_ops['accuracy'] = accuracy
    # tf.summary.scalar('accuracy', accuracy[1])

    return metric_ops
