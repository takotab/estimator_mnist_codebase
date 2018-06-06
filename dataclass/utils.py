import os
from . import reader


def import_config():
    import sys
    parentPath = os.path.abspath("..")

    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    from config import config

    return config


def get_linereader(use_validation_set, params):

    if params is None or params.train_reader is None:  # first call
        config = import_config()
        if 'restart' not in params:
            params.restart = True

        params.val_reader = reader.Reader(
            params.restart, config["MNIST"]["test"]["csv"])
        params.train_reader = reader.Reader(
            params.restart, config["MNIST"]["train"]["csv"])

    if use_validation_set:
        return params.val_reader
    else:
        return params.train_reader
