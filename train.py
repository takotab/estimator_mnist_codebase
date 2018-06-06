import utils
import argparse
import tensorflow as tf
import os
from dataclass import mnist_data
from modelclass import model
from metrics import extra_metrics
from dataclass import data
from evalute import evaluate


CONFIG = utils.import_config()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--logdir", type=str, default="./training_results/",
                        help="The path to the directory where models and metrics should be logged.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The number of datapoint used in one gradient descent step.")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001, help="The learning rate for training.")
    parser.add_argument("--dropout", type=float,
                        default=0.5, help="The dropout percentage to keep. 0 is no dropout.")
    parser.add_argument("--simple", type=bool,
                        default=True, help="Whether to use a simple dnn.")
    parser.add_argument("--baseline", type=bool,
                        default=True, help="Whether to use a baseline.")
    parser.add_argument("--deepwide", type=bool,
                        default=True, help="Whether to use a deepwide network.")
    parser.add_argument("--custom", type=bool,
                        default=True, help="Whether to use the custom network.")
    params = parser.parse_args()

    dataset_name = mnist_data.download()
    params.logdir = os.path.join(params.logdir, dataset_name)
    params.train_reader = None
    params.val_reader = None

    # the number of extra wide features. Currently (pseudo) randomly generatate values between [0, 1).
    params.extra_wide_features = 25

    # Feature columns describe how to use the input.
    my_feature_columns = []

    sizes = {
        "Deep": [28 * 28],
        "Wide": [params.extra_wide_features]
    }

    print(sizes)

    for key in sizes:
        my_feature_columns.append(
            tf.feature_column.numeric_column(key=key, shape=sizes[key]))

    params.feature_columns = my_feature_columns

    classifiers = []
    if params.baseline:
        baseline = tf.estimator.BaselineClassifier(model_dir=params.logdir + "/BaselineClassifier",
                                                   n_classes=10)
        classifiers.append((baseline, 10))

    if params.simple:
        simple = tf.estimator.DNNClassifier(hidden_units=[300, 300],
                                            feature_columns=[
            my_feature_columns[0]],
            model_dir=params.logdir +
            "/simple_model_300x300",
            n_classes=10)
        classifiers.append((simple, 10000))

    if params.deepwide:
        deepwide = tf.estimator.DNNLinearCombinedClassifier(model_dir=params.logdir + "/deep_wide_model_300x300",
                                                            linear_feature_columns=[
                                                                my_feature_columns[1]],
                                                            dnn_feature_columns=[
                                                                my_feature_columns[0]],
                                                            dnn_hidden_units=[
                                                                300, 300],
                                                            dnn_dropout=0.5,
                                                            n_classes=10
                                                            )
        classifiers.append((deepwide, 1000))

    if params.custom:
        custom = tf.estimator.Estimator(model_fn=model.model_fn,
                                        model_dir=params.logdir + "/custom_model_fn",
                                        params=params)
        classifiers.append((custom, 20000))

    tf.logging.set_verbosity('INFO')

    for classifier, max_steps in classifiers:
        classifier = tf.contrib.estimator.add_metrics(  # pylint: ignore
            classifier,
            extra_metrics
        )

        train_spec = tf.estimator.TrainSpec(input_fn=lambda: data.input_fn(
            eval=False, use_validation_set=False, params=params),
            max_steps=max_steps)

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data.input_fn(
            eval=True, use_validation_set=True, params=params),
            throttle_secs=60*10,
            start_delay_secs=60*5)

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

        # evaluate(classifier, params, result_dir='results.json')
