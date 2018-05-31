import copy
import os
import tensorflow as tf


def predict_all(estimator, input_fn, result_dir=None):
    """
    Get the predictions based on the features given by the input_fn.


    The goal here was to make something that would make a folder to showall the instances where
    the model went wrong. Unfortunatly this does not work with the current version of tensorflow.


    """

    if result_dir is None:
        result_dir = "results.txt"
    f_w = open(result_dir, "w")
    y_hat = []
    predictions = estimator.predict(input_fn, yield_single_examples=False)
    for prediction in predictions:
        for i in range(prediction["classes"].shape[0]):
            y_ = prediction["classes"][i]
            y_hat.append(int(y_))
            f_w.write(str(y_) + "\n")

    return y_hat, result_dir

# new idea put the 3 things in on file


def get_true_labels(input_fn, y_star_dir=None):
    if y_star_dir is None:
        y_star_dir = "true_labels.txt"

    f_w = open(y_star_dir, "w")
    for line in f_w.readline():
        f_w.write(line.split(',')[0] + "\n")

    return y_star, y_star_dir


def evaluator(output_fn=None, labels=None, predictions=None, labels_dir=None, predictions_dir=None):
    if output_fn is not None:
        assert callable(output_fn), "output_fn must be a function"

    labels = get_list_of_result(labels, labels_dir)
    predictions = get_list_of_result(predictions, predictions_dir)
    conf_matrix(labels, predictions)

    for label, prediction in zip(labels, predictions):
        if label is not prediction:
            if output_fn is not None:
                output_fn(label, prediction)


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


def get_list_of_result(results, result_dir):
    if type(results) is list:
        return results
    elif os.path.isfile(result_dir):
        with open(result_dir, "r") as f:
            return [int(l) for l in f.readline]
    else:
        raise Exception()


def test_output_fn(labels, predictions):
    print(labels, "is not", predictions)


if __name__ is "__main__":
    from dataclass import data
    import argparse
    from dataclass import mnist_data

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
    y_hat, y_hat_dir = predict_all(simple, input_fn=lambda: data.input_fn(
        eval=True, use_validation_set=True, params=params))
    y_star, y_star_dir = get_true_labels(input_fn=lambda: data.input_fn(
        eval=True, use_validation_set=True, params=params))
    evaluator(test_output_fn, y_star, y_hat)
