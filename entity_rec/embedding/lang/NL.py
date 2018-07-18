import os
import pickle

import config

try:
    dir = config.PREPERATION_DIR
except:
    dir = "./preperation_files/"

nl_embedding_pckl = os.path.join(dir, 'nl-embedding.pckl')
print("NL embedding", nl_embedding_pckl)
with open(nl_embedding_pckl, 'rb') as f:
    EMB_ARRAY, INT2STR, STR2INT = pickle.load(f)


def return_variables():
    return EMB_ARRAY, INT2STR, STR2INT
