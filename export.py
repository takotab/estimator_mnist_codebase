import tensorflow as tf


def export_esitimator(estimator, servable_model_dir = "./serving_savemodel",
                      **kwargs):
    # I step
    feature_columns = []
    for key in kwargs:
        feature_columns += kwargs[key]

    # II step
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

    # III step
    export_input_fn = \
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)

    # IV step

    servable_model_path = estimator.export_savedmodel(servable_model_dir,
                                                      export_input_fn)

    return servable_model_path
