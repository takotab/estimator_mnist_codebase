import os


def import_config():
    import sys
    # dir = os.sep.join(os.getcwd().split(os.sep)[:-1])
    parentPath = os.path.abspath("..")

    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    from config import config

    return config


def get_linereader(use_validation_set, params):

    if params.train_reader is None:  # first call
        config = import_config()
        params.val_reader = open(config["MNIST"]["test"]["csv"], "r")
        params.train_reader = open(config["MNIST"]["train"]["csv"], "r")

    if use_validation_set:
        return params.val_reader
    else:
        return params.train_reader
