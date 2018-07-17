import config
import pickle
import os

nl_embedding_pckl = os.path.join(config.PREPERATION_DIR, 'nl-embedding.pckl')
print("NL embedding", nl_embedding_pckl)
with open(nl_embedding_pckl, 'rb') as f:
    EMB_ARRAY, INT2STR, STR2INT = pickle.load(f)


def return_variables():
    return EMB_ARRAY, INT2STR, STR2INT
