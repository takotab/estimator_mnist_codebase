from scipy.spatial.distance import cosine
import numpy as np
from . import util
from entity_rec.embedding import Language as Language


def match_text(value, options, language=None):
    if language is None:
        language = Language()

    value = value.replace("\"", "")
    value = value_extras(value)
    value_emb = language.get_sentence_embedding(value)
    value_emb = util.emb_average(value_emb)

    best = {"choice": None, "confidence": 0}
    for i, option in enumerate(options):
        option = option.replace("\"", "")
        if value == option:
            best = {"choice": i, "confidence": 1}
            print("found a direct match:", best, options, value)
            return best

        option_emb = language.get_sentence_embedding(option)
        option_emb = util.emb_average(option_emb)

        sim = 1 - cosine(value_emb, option_emb)
        print(option, sim)
        if sim > best["confidence"]:
            best["confidence"] = sim
            best["choice"] = i

    if best["confidence"] > 0.9:
        print("found a match:", best, options, value)

    return best


def value_extras(value):
    """
    made to add some extra words to value.
    make the classification better.
    """
    if value == "Neen":
        value = ["Neen nee niet"]

    return value
