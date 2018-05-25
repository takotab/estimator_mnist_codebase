import os


def import_config():
    import sys
    # dir = os.sep.join(os.getcwd().split(os.sep)[:-1])
    parentPath = os.path.abspath("..")

    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    from config import config

    return config
