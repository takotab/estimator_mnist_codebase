import numpy as np

def emb_average(emb_list):
    return np.max(emb_list, axis=0)
