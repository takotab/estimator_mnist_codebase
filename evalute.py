import datetime
import copy
import os
import tensorflow as tf
from dataclass import data, reader, mnist_data
from argparse import Namespace
import utils
import argparse
CONFIG = utils.import_config()


def evaluate(classifier, params, eval_dir=None, result_dir=None, wrong_fn=None, save_all=False):
    """
    Get the predictions based on the features given by the input_fn.

    The goal here was to make something that would make a folder to showall the instances where
    the model went wrong. Unfortunatly this does not work with the current version of tensorflow.


    """
    assert result_dir is not None or wrong_fn is not None

    if eval_dir is None:
        eval_dir = CONFIG["MNIST"]["test"]["csv"]

    json_dict = {}

    i = 0
    start_time = datetime.datetime.now()
    for features, labels in get_eval_data(params, eval_dir):
        predictions = classifier.predict(
            input_fn=lambda: data.eval_input_fn(
                features, labels, params.batch_size),
            yield_single_examples=False)
        for prediction in predictions:

            if result_dir is not None:

                json_dict = save_result(
                    json_dict, features, labels, prediction["classes"], save_all)

            if wrong_fn is not None:
                wrong_fn(labels=labels,
                         predictions=prediction["classes"], save_all=save_all)
        i += 1
        print("Iteration: {} datasample: {} running time: {}".format(i, i *
                                                                     params.batch_size, str(datetime.datetime.now() - start_time)))

    import json
    if result_dir is not None:
        with open(result_dir, "w") as f:
            json.dump(json_dict, f)

    return None


def get_eval_data(params, eval_dir):

    line_reader = reader.Reader(False, eval_dir)
    while True:
        try:
            deep, wide, label = data.get_batch(eval, params, line_reader)
            yield {"Deep": deep, "Wide": wide}, label
        except StopIteration:
            raise StopIteration


def conf_matrix(labels, predictions, str_labels=None):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    if str_labels is None:
        str_labels = [str(i) for i in range(max(labels))]

    cm = confusion_matrix(labels, predictions, str_labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("conf_matrix.png")


def test_output_fn(labels, predictions):
    print(labels, "is not", predictions)


def save_result(json_dict, features, labels, predictions, save_all):
    # initalize json
    if "features" not in json_dict:
        json_dict["features"] = {}
        for key in features:
            json_dict["features"][key] = []
        json_dict["label"] = []
        json_dict["prediction"] = []

    # fill the dam thing
    for i in range(labels.shape[0]):

        if save_all or int(labels[i, 0]) is not int(predictions[i]):
            for key in features:
                json_dict["features"][key].append(list(features[key][i, :]))
            json_dict['label'].append(labels[i, 0])
            json_dict['prediction'].append(int(predictions[i]))

    return json_dict


if __name__ is "__main__":
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

    my_feature_columns = []

    sizes = {
        "Deep": [28 * 28],
        "Wide": [params.extra_wide_features]
    }

    print(sizes)

    for key in sizes:
        my_feature_columns.append(
            tf.feature_column.numeric_column(key=key, shape=sizes[key]))

    simple = tf.estimator.DNNClassifier(feature_columns=[
        my_feature_columns[0]],
        # warm_start_from="./trainings_results/MNIST_estimator/simple_model_300x300/.",
        model_dir=r".\training_results\MNIST_estimator\simple_model_300x300",
        hidden_units=[300, 300],
        n_classes=10)
    assert simple.latest_checkpoint() is not None, "did not found model"

    params.restart = False
    evaluate(simple, params, result_dir='results.txt')
