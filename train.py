import utils
import argparse
import tensorflow as tf
import os
from dataclass import data
from modelclass import model
from metrics import f_score_tf


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn, name):
        self.estimator = estimator
        self.input_fn = input_fn
        self.name = name

    def after_save(self, session, global_step):
        print("RUNNING EVAL: {}".format(self.name))
        self.estimator.evaluate(self.input_fn, name=self.name)
        print("FINISHED EVAL: {}".format(self.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--logdir", type=str, default="./training_results/estimator",
                        help="The path to the directory where models and metrics should be logged.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The number of datapoint used in one gradient descent step.")
    parser.add_argument("--max_words", type=int, default=100,
                        help="The maximal number of words in sentence.")
    parser.add_argument("--num_units", type=int, default=8,
                        help="The number of units in the lstm layer.")
    parser.add_argument("--num_layers", type=int,
                        default=3, help="The number of layers of lstm.")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001, help="The learning rate for training.")
    parser.add_argument("--bw_lstm", type=bool,
                        default=False, help="Whether to use backwards lstm cells.")
    parser.add_argument("--fw_lstm", type=bool,
                        default=False, help="Whether to use forward lstm cells.")
    parser.add_argument("--dropout", type=float,
                        default=0.5, help="The dropout percentage to keep. 0 is no dropout.")
    parser.add_argument("--simple", type=bool,
                        default=False, help="Whether to use a simple dnn.")
    params = vars(parser.parse_args())

    params["dense_size"] = data.STATIC_DS["dense_size"]
    params["sparse_size"] = data.STATIC_DS["sparse_size"]
    params["num_entities"] = data.STATIC_DS["num_entities"]
    params["entities"] = data.STATIC_DS["entities"]
    # Feature columns describe how to use the input.
    my_feature_columns = []

    if params["simple"]:
        params["max_words"] = 3
        sizes = {
            "Dense": [params["max_words"] * data.STATIC_DS["dense_size"]],
            "Sparse": [params["max_words"] * data.STATIC_DS["sparse_size"]],
            "seq_len": [1],
            "label": [1]
        }
    else:
        sizes = {
            "Dense": [params["max_words"], data.STATIC_DS["dense_size"]],
            "Sparse": [params["max_words"], data.STATIC_DS["sparse_size"]],
            "seq_len": [1],
            "label": [params["max_words"]]
        }
    print(sizes)
    for key in sizes:
        my_feature_columns.append(
            tf.feature_column.numeric_column(key=key, shape=sizes[key]))

    params['feature_columns'] = my_feature_columns
    # estimator = tf.estimator.LinearClassifier(
    #     feature_columns=my_feature_columns)

    if params["simple"]:
        estimator = tf.estimator.DNNClassifier([params["num_units"]] * params["num_layers"],
                                               my_feature_columns[:-2],
                                               model_dir=params["logdir"] +
                                               "/simple_model",
                                               n_classes=9)
        # estimator = tf.estimator.BaselineClassifier(
        #     model_dir=params["logdir"] +
        #     "/BaselineClassifier",
        #     n_classes=9)
    else:
        estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                           model_dir=params["logdir"] + "/" +
                                           utils.make_name(params),
                                           params=params)

    tf.logging.set_verbosity('INFO')

    estimator = tf.contrib.estimator.add_metrics(  # pylint: ignore
        estimator,
        f_score_tf
    )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: data.input_fn(
        eval=False, use_validation_set=False, params=params),
        max_steps=20000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data.input_fn(
        eval=True, use_validation_set=True, params=params),
        throttle_secs=60*2,
        start_delay_secs=60*3)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    results = estimator.evaluate(input_fn=data.input_fn(
        eval=True, use_validation_set=True, params=params))
    print(results)
