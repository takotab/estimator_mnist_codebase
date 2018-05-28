
import datetime
from dataclass import mnist_data
import tensorflow as tf
from dataclass import data


def run_over_data(params, fn_to_get_batch, name="trainingset", use_validation_set=False, func=None):
    print(name, "start test")
    start_time = datetime.datetime.now()
    if use_validation_set:
        fil_len = 10000
    else:
        fil_len = 60000

    total = int(fil_len/params.batch_size)+5

    with tf.Session() as sess:
        for i in range(total):
            if params.use_tf:
                _batch = sess.run(
                    fn_to_get_batch(use_validation_set, use_validation_set, params))
            else:
                _batch = fn_to_get_batch(
                    use_validation_set, params)
                assert _batch[0]["Deep"].shape[1] is 748

            if func is not None:
                func(_batch, params.use_tf)

            if i % 5 is 0 and i is not 0:
                run_time = datetime.datetime.now() - start_time
                print(i, total, run_time/i, "ETA",
                      start_time + (run_time/i)*total)

    print(name, "DONE")


if __name__ == "__main__":
    mnist_data.download()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./training_results/estimator",
                        help="The path to the directory where models and metrics should be logged.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The number of datapoint used in one gradient descent step.")

    params = parser.parse_args()

    params.use_tf = True
    params.train_reader = None
    params.val_reader = None
    params.extra_wide_features = 10
    fn_to_get_batch = data.input_fn
    # verify generator output
    run_over_data(params, fn_to_get_batch, name="trainingset",
                  use_validation_set=False, func=None)

    run_over_data(params, fn_to_get_batch, name="validationset",
                  use_validation_set=True, func=None)
