import tensorflow as tf


def export_esitimator(estimator, params, serving_model_dir = "./serving_savemodel", ):
    # I step
    feature_columns = params

    # II step
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

    # III step
    export_input_fn = \
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
                feature_spec)

    # IV step

    serving_model_path = estimator.export_savedmodel(serving_model_dir,
                                                     export_input_fn)

    return serving_model_path
