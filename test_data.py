
import datetime
from dataclass import mnist_data
import tensorflow as tf
from dataclass import data


def run_over_data(params, fn_to_get_batch, name="trainingset", use_validation_set=False, func=None):
    print(name, "start test")
    start_time = datetime.datetime.now()
    total = 60000
    params.restart = False
    with tf.Session() as sess:
        i = 0
        while True:
            if params.use_tf:
                _batch = sess.run(
                    fn_to_get_batch(use_validation_set, use_validation_set, params))
            else:
                _batch = fn_to_get_batch(
                    use_validation_set,  params)
                # assert _batch[0]["Deep"].shape[1] is 748
            if _batch is StopIteration:
                break

            if func is not None:
                func(_batch, params.use_tf)

            if i % 50 is 0 and i is not 0:
                run_time = datetime.datetime.now() - start_time
                print(i, total, run_time/i, "ETA",
                      start_time + (run_time/i)*total)

            i += 1
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

    params.use_tf = False
    params.train_reader = None
    params.val_reader = None
    params.extra_wide_features = 10
    fn_to_get_batch = data.get_batch
    # verify generator output
    run_over_data(params, fn_to_get_batch, name="trainingset",
                  use_validation_set=True, func=None)

    fn_to_get_batch = data.get_batch
    run_over_data(params, fn_to_get_batch, name="validationset",
                  use_validation_set=True, func=None)
