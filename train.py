import argparse
import os

import tensorflow as tf

import utils
from dataclass import data
from entity_rec import car_entities
from export import export_esitimator
from metrics import extra_metrics
from modelclass import model

CONFIG = utils.import_config()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--logdir", type = str, default = "./training_results/",
                        help = "The path to the directory where models and metrics should be "
                               "logged.")
    parser.add_argument("--batch_size", type = int, default = 32,
                        help = "The number of datapoint used in one gradient descent step.")
    parser.add_argument("--learning_rate", type = float,
                        default = 0.001, help = "The learning rate for training.")
    parser.add_argument("--dropout", type = float,
                        default = 0.5, help = "The dropout percentage to keep. 0 is no dropout.")
    parser.add_argument("--simple", type = bool,
                        default = True, help = "Whether to use a simple dnn.")
    parser.add_argument("--baseline", type = bool,
                        default = False, help = "Whether to use a baseline.")
    parser.add_argument("--deepwide", type = bool,
                        default = False, help = "Whether to use a deep-wide network.")
    parser.add_argument("--custom", type = bool,
                        default = False,
                        help = "Whether to use the custom convectional network.")
    params = parser.parse_args()

    # dataset_name = mnist_data.download()
    params.data_dir_train = os.path.join(os.getcwd(), "small_example_car_data.csv")
    dataset_name = "test_nlp"
    params.logdir = os.path.join(params.logdir, dataset_name)
    params.train_reader = None
    params.val_reader = None

    sizes = car_entities.FEATURE_INFO

    print(sizes)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in sizes:
        my_feature_columns.append(
                tf.feature_column.numeric_column(key = key, shape = sizes[key]))

    params.feature_columns = my_feature_columns

    classifiers = []
    if params.baseline:
        baseline = tf.estimator.BaselineClassifier(
                model_dir = params.logdir + "/BaselineClassifier",
                n_classes = car_entities.NUM_CLASSES)
        classifiers.append((baseline, 10))

    if params.simple:
        simple = tf.estimator.DNNClassifier(hidden_units = [50, 10],
                                            feature_columns = [
                                                my_feature_columns[0]],
                                            model_dir = params.logdir +
                                                        "/simple_model_50x10",
                                            n_classes = car_entities.NUM_CLASSES,
                                            )
        classifiers.append((simple, 50))

    if params.deepwide:
        deepwide = tf.estimator.DNNLinearCombinedClassifier(
                model_dir = params.logdir + "/deep_wide_model_300x300",
                linear_feature_columns = [
                    my_feature_columns[1]],
                dnn_feature_columns = [
                    my_feature_columns[0]],
                dnn_hidden_units = [
                    300, 300],
                dnn_dropout = 0.5,
                n_classes = car_entities.NUM_CLASSES
                )
        classifiers.append((deepwide, 1000))

    if params.custom:
        custom = tf.estimator.Estimator(model_fn = model.model_fn,
                                        model_dir = params.logdir + "/custom_model_fn",
                                        params = params)
        classifiers.append((custom, 200))

    tf.logging.set_verbosity('INFO')

    for classifier, max_steps in classifiers:
        classifier = tf.contrib.estimator.add_metrics(  # pylint: ignore
                classifier,
                extra_metrics
                )

        train_spec = tf.estimator.TrainSpec(input_fn = lambda: data.input_fn(
                eval = False, use_validation_set = False, params = params),
                                            max_steps = max_steps)

        eval_spec = tf.estimator.EvalSpec(input_fn = lambda: data.input_fn(
                eval = True, use_validation_set = False, params = params),
                                          throttle_secs = 60 * 10,
                                          start_delay_secs = 60 * 5)

        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

        export_esitimator(classifier, sizes)
    # evaluate(classifier, params, result_dir='results.json')
