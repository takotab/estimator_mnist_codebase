import random
import numpy as np
import tensorflow as tf
from . import utils
CONFIG = utils.import_config()


def input_fn(eval, use_validation_set, params):
    """Outputs a tuple containing a features tensor and a labels tensor

    Args:
      eval: whether we are evaluating or training. Training generates episodes indefinitely, evaluating generates params.batches_per_evaluation.
      use_validation_set: bool, if True then load the validation set, otherwise load the training set.
      params: a dictionary of configuration parameters

    Returns:
      (features_t, labels_t)
    """

    ds = tf.data.Dataset.from_generator(generator=lambda: episode_batch_generator(use_validation_set, params),

                                        output_types=(
                                            tf.float32, tf.float32, tf.int32)
                                        )
    if eval:
        ds = ds.take(CONFIG["MNIST"]["test"]["size"])

    # 4 is arbitrary, a little prefetching helps speed things up
    ds = ds.prefetch(100)

    deep_t, wide_t, labels_t = ds.make_one_shot_iterator().get_next()

    return {"Deep": deep_t, "Wide": wide_t}, labels_t


def episode_batch_generator(use_validation_set, params):
    """Yields a batch of episode images and their corresponding labels

    Args:
        use_validation_set: bool, whether to use the train or validation dataset
        params: application level constants

    Yields:
        A tuple (features, labels)
        * A dict of features:
            * "Deep": (batch_size,28*28), np.float32
            * "Wide": (batch_size, extra_wide_features), np.float32
        * labels: (batch_size, max_words), np.int32
    """
    while True:
        yield get_batch(use_validation_set, params)


def get_batch(use_validation_set, params):
    """A batch of params.batch_size with datapoints
    The reason I did not use the recomened way with tf.data.FixedLengthRecordDataset
    is that I think this is much more clear what is going happening. And it is easier to adapt to other needs.

    Args:
        use_validation_set: bool, whether to use the train or validation dataset
        params: application level constants

    Returns:
        A tuple (features, labels)
            * A dict of features:
                * "Dense": (batch_size, max_words, dense_size), np.float32
                * "Sparse": (batch_size, max_words, sparse_size), np.float32
            * labels: (batch_size, max_words), np.int32
    """

    line_reader = utils.get_linereader(
        use_validation_set, params)  # returns a IO textwrapper

    labels, deep, wide = [], [], []

    while len(labels) < params.batch_size:

        # TODO extra bullshit class
        line = line_reader.readline().replace("\n", "").split(",")
        labels.append(int(line[0]))
        deep.append(np.float(line[1:]))
        wide.append(np.random.rand(params.extra_wide_features))

    assert len(labels) == len(deep) == len(
        wide), "the features/labels do not have the same datapoints in a batch"

    return np.array(deep),  np.array(wide),  np.array(labels)
