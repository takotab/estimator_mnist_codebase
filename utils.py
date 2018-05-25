import os
import numpy as np
import glob
import pickle
from sklearn.metrics import f1_score
import tensorflow as tf


def import_config():
    import sys
    # dir = os.sep.join(os.getcwd().split(os.sep)[:-1])
    parentPath = os.path.abspath("..")

    if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
    from config import config

    return config


def pickle_save(obj, name):
    pickle.dump(obj, open(os.path.join("data", "temp_save", name), "wb"))


def file_len(fname):
    with open(fname, 'rb') as f:
        i = 0
        for _ in f:
            i += 1
    return i + 1


def get_f_score(mdl, train):
    y_pred_total = []
    for y_, seq_len_, X_ in train:
        _, y_pred = mdl.sess.run([mdl.update_op_group, mdl.softmax], feed_dict={mdl.X: X_[:, :, :-mdl.static_ds["sparse_size"]],
                                                                                mdl.sparse: X_[:, :, -mdl.static_ds["sparse_size"]:],
                                                                                mdl.sequence_length: seq_len_,
                                                                                mdl.y: y_,
                                                                                mdl.keep_prob: mdl.static_ds["keep_prob"]})
        y_pred_total.append(y_pred)
    f_score_tf = mdl.sess.run(mdl.f_score)
    pickle_save([y_pred_total, f_score_tf], "tf_f1_score")
    pass

    # _, y_tf = mdl.predict(x=X_, seq_len=seq_len_)
    # f1_score_sk = []
    # y_reshape = np.ndarray.flatten(y_)
    # y_tf_reshape = np.ndarray.flatten(y_tf)
    # f1_score_sk.append(f1_score(y_reshape, y_tf_reshape, average='macro'))
    # f1_score_sk.append(f1_score(y_reshape, y_tf_reshape, average='micro'))
    # f1_score_sk.append(
    #     f1_score(y_reshape, y_tf_reshape, average='weighted'))
    # f1_score_sk.append(f1_score(y_reshape, y_tf_reshape, average=None))
    # print(f1_score_sk, f1_score_tf)
    # total.append([*f1_score_sk, f1_score_tf])
import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def make_name(params):
    # basic way of making a name

    name = "model"
    exculde = ["entities", 'feature_columns', "logdir"]
    for key in params:
        if key not in exculde:
            name += "__" + key + "_" + str(params[key])
    print("name", name)
    return name


import datetime


def run_over_data(params, input_fn, name="trainingset", use_validation_set=False, func=None):
    print(name, "start test")
    start_time = datetime.datetime.now()
    total = 60
    # otherwise something will go different in data.py
    params["simple"] = False

    with tf.Session() as sess:
        for i in range(total):
            _batch = sess.run(
                input_fn(use_validation_set, use_validation_set, params))

            if func is not None:
                func(_batch)

            if i % 5 is 0 and i is not 0:
                run_time = datetime.datetime.now() - start_time
                print(i, total, run_time/i, "ETA",
                      start_time + (run_time/i)*total)

    print(name, "DONE")
