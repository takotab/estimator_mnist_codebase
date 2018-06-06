import hashlib
HASH_DICT = {}


def add_datapoint(x, y):
    print(x)
    hash_datapoint = hashlib.sha1(x)
