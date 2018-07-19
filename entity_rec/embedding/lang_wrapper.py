import nltk
import numpy as np
from nltk.tokenize import wordpunct_tokenize

nltk.download('stopwords')


class Language:

    def __init__(self, lang = "NL"):
        if lang == "NL":
            from .lang import NL as lang
        else:
            print("NOT YET SUPPORTED")
            assert lang == "NL", "OTHER LANG NOT YET SUPPORTED"
        self.emb_array, self.int2str, self.str2int = lang.return_variables()

    def word2emb(self, word):
        if not word in self.str2int:
            if word.lower() in self.str2int:
                word = word.lower()
            else:
                return np.zeros(self.emb_array.shape[1])

        _int = self.str2int[word]
        return self.emb_array[_int, :]

    def get_sentence_embedding(self, sentence, sentence_dict = False):
        assert type(sentence) is str
        sentence_dictonary = {}
        sentence_embedding = []
        sentence = [word for word in wordpunct_tokenize(sentence)]

        for i, word in enumerate(sentence):
            sentence_dictonary[i] = word
            emb = self.word2emb(word)
            sentence_embedding.append(emb)

        sentence_embedding = np.array(sentence_embedding)
        if sentence_dict:
            return sentence_embedding, sentence_dictonary
        return sentence_embedding
