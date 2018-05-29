import os


def import_config():
    import sys
    parentPath = os.path.abspath("..")

    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    from config import config

    return config


def get_linereader(use_validation_set, params):

    if params.train_reader is None:  # first call
        config = import_config()
        params.val_reader = reader(config["MNIST"]["test"]["csv"])
        params.train_reader = reader(config["MNIST"]["train"]["csv"])

    if use_validation_set:
        return params.val_reader
    else:
        return params.train_reader


class reader:
    """
    Class that reads the data and restarts if it reaches the end.
    """

    def __init__(self, filedir):
        self.filedir = filedir
        self.f = open(filedir, "r")

    def next(self):
        line = self.f.readline()
        if not line:
            self.f.close()
            self.f = open(self.filedir, "r")
            line = self.f.readline()
        return(line)
