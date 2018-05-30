import tensorflow as tf
from dataclass import data


def show_errors(estimator, test_loc, error_loc=None):
    estimator


if __name__ is "__main__":
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
        my_feature_columns[0]], warm_start_from="./trainings_results/MNIST_estimator/simple_model_300x300/.",
        hidden_units=[300, 300],
        n_classes=10)

    show_errors(simple, test_loc)
