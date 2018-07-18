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

        if 'restart' not in params:
            params.restart = True
        if use_validation_set:
            params.val_reader = reader.Reader(
                    params.restart,
                    params["data_dir"]["val"],
                    )
        params.train_reader = reader.Reader(
                params.restart,
                params["data_dir"]["train"],
                )

    if use_validation_set:
        return params.val_reader
    else:
        return params.train_reader
