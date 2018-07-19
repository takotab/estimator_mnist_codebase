import numpy as np
import tensorflow as tf

from entity_rec import sub_embedding
from entity_rec.embedding import Language

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
    "Audi" : 0,
    }

# max_words = 50
FEATURE_INFO = {
    "Deep": [300],  # 300 = size of embedding
    "Wide": [len(CAR_SUB_DICT)],
    }
for key in FEATURE_INFO:
    FEATURE_INFO[key] = tf.feature_column.numeric_column(key = key, shape = FEATURE_INFO[key])


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
    sentence_emb = np.median(sentence_emb, axis = 0)
    # total_emb = np.concatenate((sentence_emb, sub_word), axis = 1)

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


# actually a test
def test_make_features():
    text = ["ik wil de Opel Corsa",
            "ik wil de Ford fiesta",
            "ik wil de Opel Astra",
            "ik wil de Volvo V60",
            "ik wil de Tesla model S",
            "ik wil de BMW 3 Series",
            "ik wil de Ford focus",
            "ik wil de Audi A3",
            ]
    for t in text:
        print(t)
        features = make_features(t)
        for key in features:
            print(key, features[key].shape)
            if key is "Wide":
                print(features[key])
