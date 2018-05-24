from dataclass.data import get_batch
from dataclass.data import input_fn
import datetime
from tests.utils import from_batch_to_dict_datapoint
from tests.run_over_data import run_over_data
from tests.details_data import Details_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./training_results/estimator",
                        help="The path to the directory where models and metrics should be logged.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The number of datapoint used in one gradient descent step.")
    parser.add_argument("--max_words", type=int, default=100,
                        help="The maximal number of words in sentence.")
    parser.add_argument("--use_tf", type=bool, default=False,
                        help="this way it is easier to find and fix bugs")

    params = parser.parse_args()
    params = vars(parser.parse_args())
    params["simple"] = False
    if params["simple"]:
        params["max_words"] = 3

    # this way it is easier to find and fix bugs
    if params["use_tf"]:
        fn_to_get_batch = input_fn
    else:
        fn_to_get_batch = get_batch
    details_data = Details_data()

    def fn(x): return details_data.add(from_batch_to_dict_datapoint(x))
    # verify generator output
    run_over_data(params, fn_to_get_batch, name="trainingset",
                  use_validation_set=False, func=fn)

    details_data.give_details("trainingset")

    del(fn)
    details_data = Details_data()

    def fn(x): return details_data.add(from_batch_to_dict_datapoint(x))

    run_over_data(params, fn_to_get_batch, name="validationset",
                  use_validation_set=True, func=fn)

    details_data.give_details("validationset")
