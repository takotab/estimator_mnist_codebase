import config
import tensorflow as tf
import numpy as np
from .chatbot import dialog
from . import procces_data
from . import check_prep_files
from .match import *
check_prep_files.init()

# Please note I rewrote this thing but only without intent yet so hopefully this will be replaced
# till that time please do not try to understand dialog_ and procces they are awefull

dialog_ = dialog(word2vec=True, lang='NL', max_words=100)
procces = procces_data.prep_data(dialog_class=dialog_, chance_on_name=0)

SESS = tf.Session()
path = config.WEIGHTS_DIR  # , "new_data_0.1.60.0001, 1e-05_0.5")
print(path)
latest_check = tf.train.latest_checkpoint(path)
print(latest_check)
new_saver = tf.train.import_meta_graph(
    latest_check + ".meta", clear_devices=True)
print(new_saver, SESS)
new_saver.restore(SESS, latest_check)

GINFO = {"procces": procces,
         "logits": tf.get_collection("output_logits")[0],
         "X": tf.get_collection("X")[0],
         "s_sequence_length": tf.get_collection("s_sequence_length")[0],
         "keep_prob": tf.get_collection("keep_prob")[0],
         "pred_goal": tf.get_collection("pred_goal")[0],
         "extra_info_place": tf.get_collection("extra_info_place")[0]}


def predict(text):

    return_dict = _tfInference(text)
    return return_dict


def _tfInference(text):
    extra_info_num_of_sparce_e = 42
    ginfo = GINFO
    # print(ginfo)
 # feel free to adapt the conv enty and the first name (info_1) and last name (info_2) you want it to return
    # with making the datapoints I do not use the dict which start with _ for making identifing entities
    seq_len, x_, sens_dict = ginfo['procces'].make_datapoint({"_conv": text})
    # print(tf.shape(ginfo["extra_info_place"]))
    # run the model with the datapoint generated
    feed_dict = {ginfo["X"]: x_[:, :, :-extra_info_num_of_sparce_e],
                 ginfo["extra_info_place"]: x_[:, :, -extra_info_num_of_sparce_e:],
                 ginfo["s_sequence_length"]: seq_len,
                 ginfo["keep_prob"]: 1}

    # print(feed_dict)
    # logits, pred_goal zijn de end-point tensors van het model
    y_, g_ = SESS.run([ginfo['logits'], ginfo['pred_goal']],
                      feed_dict=feed_dict)

    # y_ is the entity
    # g_ is the intent

#     print(np.round(y_,1))
    # print(g_.shape)
    return ginfo['procces'].get_info(y=y_, x=x_, seq_len=seq_len, intent=g_, sen_dict=sens_dict, plot=False)
#     y_output_model = y_
