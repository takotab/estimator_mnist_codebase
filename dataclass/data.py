import numpy as np
import tensorflow as tf

from entity_rec import car_entities
from entity_rec.embedding import Language
from . import utils  # pylint : ignore

CONFIG = utils.import_config()
NEDERLANDS = Language()


def input_fn(eval, use_validation_set, params):
    """Outputs a tuple containing a features tensor and a labels tensor

    Args:
      eval: whether we are evaluating or training. Training generates episodes indefinitely,
      evaluating generates params.batches_per_evaluation.
      use_validation_set: bool, if True then load the validation set, otherwise load the training
      set.
      params: a dictionary of configuration parameters

    Returns:
      (features_t, labels_t)
    """

    ds = tf.data.Dataset.from_generator(
            generator = lambda: episode_batch_generator(use_validation_set, params),

            output_types = (
                tf.float32, tf.float32, tf.int32)
            )
    if eval:
        pass  # TODO: eval dataset
    else:
        pass
        # ds.shuffle(60000)

    # 4 is arbitrary, a little pre-fetching helps speed things up
    # ds = ds.prefetch(100)

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


def get_batch(use_validation_set, params, line_reader = None):
    """A batch of params.batch_size with datapoints
    The reason I did not use the recomened way with tf.data.FixedLengthRecordDataset
    is that I think this is much more clear what is going happening. And it is easier to adapt to
    other needs.

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
    if line_reader is None:
        line_reader = utils.get_linereader(
                use_validation_set, params)  # returns a iter

    labels, deep, wide = [], [], []

    while len(labels) < params.batch_size:
        # get next line in csv file
        line = line_reader.next().replace("\n", "").split(",")  # default for CSV
        if line[0] is '':
            raise StopIteration()

        _labels, _deep, _wide = interpret_line(line)
        labels.append(_labels)
        deep.append(_deep)
        wide.append(_wide)
        
    assert len(labels) == len(deep) == len(
            wide), "the features/labels do not have the same datapoints in a batch"

    deep = np.array(deep, dtype = float)
    wide = np.array(wide, dtype = float)
    label = np.array(labels)[:, np.newaxis]
    return deep, wide, label


def interpret_line(line):
    """

    :param line:
        The line from the data source to be interpreted
    :return:
        data ready for the model (features and labels)
    """
    labels = line[0]
    features = car_entities.make_features(line[1], language = NEDERLANDS)
    deep = features["Deep"]
    wide = features["Wide"]
    return labels, deep, wide


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""

    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features

    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
