import tensorflow as tf
import numpy as np
from entity_rec.embedding import Language
from entity_rec import sub_embedding

CAR_INTENT = ["Opel Corsa",
              "Ford fiesta",
              "Opel Astra",
              "Volvo V60",
              "Tesla model S",
              "BMW 3 Series",
              "Ford focus",
              "Audi A3",
              ]
NUM_CLASSES = len(CAR_INTENT)
CAR_SUB_DICT = {
    "Tesla": 1,
    "Opel" : 2,
    "BMW"  : 3,
    "Ford" : 4,
    "Volvo": 5,
    "Corsa": 6,
    "Audi" : 7,
    }

max_words = 50
FEATURE_INFO = {
    "Deep": [max_words * 300],  # 300 = size of embedding
    "Wide": [max_words * len(CAR_SUB_DICT)],
    }


def make_features(sentence, language = None, params = None):
    """

    :param sentence:
    :param language:
    :param params:
    :return:
    """

    if language is None:
        language = Language()

    sentence_emb = language.get_sentence_embedding(sentence)
    sub_word = sub_embedding.add_subsentence(sentence, CAR_SUB_DICT)
    # total_emb = np.concatenate((sentence_emb,sub_word),axis = 1)

    features = {
        "Deep": sentence_emb,
        "Wide": sub_word,
        }
    return features


def predict(sentence, language = None, estimator = None):
    if estimator is None:
        car_estimator = make_estimator()

    pred_label = car_estimator.evaluate(input_fn = features, params = params)
    return CAR_INTENT[pred_label]


def make_estimator():
    # TODO: restore estimator
    raise NotImplementedError()
