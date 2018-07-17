import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
# import matplotlib.pyplot as plt
import time
# import gensim
stemmer = EnglishStemmer()
import os
import sys
import config
# sys.path.insert(0, os.path.join(os.getcwd(), "va", "swagger_server"))


def file_len(fname):
    with open(fname, 'rb') as f:
        i = 0
        for a in f:
            i += 1
    return i + 1


class dialog:
    def __init__(self, word2vec = False, max_words = 100, lang = 'NL'):

        self.lang = lang
        self.max_words = max_words
        self.input_depth = 382
        self.extra_input = 82
        self.num_unknown = 0
        self.array = np.full(
            [self.max_words, self.input_depth - self.extra_input], np.nan)
        # first 26 are the letters of alphabet then we have 4 left to signal other things
        self.add_vector = np.full([self.max_words, self.extra_input], np.nan)
        self.word2vec = word2vec
        self.alphabet = {abc: i for i, abc in enumerate(
            'abcdefghijklmnopqrstuvwxyz0123456789?')}
        self.alphabet["<UNK>"] = len(self.alphabet)
        self.alphabet["<SYN>"] = 38
        self.alphabet["<SPACE>"] = 39
        self.alphabet["<upper>"] = 40
        self.alphabet["<city>"] = 41
        self.alphabet["<f_name>"] = 42
        self.alphabet["<l_name>"] = 43
        self.alphabet["<eou>"] = 44
        self.missing_words = ["certimed",
                              "omkosten",
                              "declareer",
                              "vestapro",
                              "ziekmelden",
                              "menuoptie",
                              "betaalcodes",
                              "looninfo",
                              "datafreeze",
                              "loonsberekening",
                              "loonbrieven",
                              "belastingopgaaf",
                              "belastingopgave",
                              "jaaropgaaf",
                              "loongegevens",
                              "loonsgegevens",
                              "loonbrief",
                              "glijsaldo",
                              "recuperatieverlof",
                              "tikkingen",
                              "compensatieverlof",
                              "inhaalrust",
                              "inplan",
                              "verlofaanvraag",
                              "verlofuren",
                              "vestatijd",
                              "dienstvrijstelling",
                              "wachtweek",
                              "rmatie",
                              "beschikbaarheids",
                              "onbetaal",
                              "sduur",
                              "vervangingsdag",
                              "partime",
                              "bedraagd",
                              "annuleer",
                              "beeindig",
                              "cancellen"]
        number = 45
        for missing_word in self.missing_words:
            self.alphabet[missing_word] = number
            number += 1
            #         print(self.alphabet)
        self.r_alphabet = {
            self.alphabet[abc]: abc for abc in list(self.alphabet.keys())}
        # print(self.r_alphabet)
        if word2vec:
            if self.lang == 'EN':
                # Load Google's pre-trained Word2Vec model.
                print("loading word2vec model")
                # model = gensim.models.KeyedVectors.load_word2vec_format('./embedding/GoogleNews-vectors-negative300.bin', binary=True)
                model = None
                self.word_vectors = model.wv

                del model
            elif self.lang == 'NL':
                from .embedding.lang import NL
                # self.emb = lang_wrapper.Language("NL")
                self.emb_array = NL.EMB_ARRAY
                self.int2str = NL.INT2STR
                self.str2int = NL.STR2INT
        else:
            print()
            self.word_vectors = {abc: np.random.randn(300,) for abc in [
                "tako", "UNK", 'Greetings', 'Hello', 'agent', "you", "name", "is"]}
        #similar_by_vector(vector, topn=10, restrict_vocab=None)

    def __hash__(self):
        return hash((self.total_conversation, self.max_words, self.word2vec))

    def __eq__(self, other):

        if not isinstance(other, type(self)):
            return NotImplemented
        return self.total_conversation == other.total_conversation and self.max_words == other.max_words and self.word2vec == other.word2vec

    def get_word(self, vector):
        if self.word2vec and self.lang == 'EN':
            return self.word_vectors.similar_by_vector(vector, topn=3)
        else:
            return [("UNK", 1)]

        #self.add_vector += self.make_ad
    def check_in_dict(self, word):
        if self.word2vec and self.lang == 'EN':
            return word in self.word_vectors.vocab
        elif self.word2vec and self.lang == 'NL':
            if not word in self.str2int:
                return word.lower() in self.str2int
            return True
        else:
            return word in self.word_vectors

    def str2emb(self, word):
        if not word in self.str2int:
            word = word.lower()
        _int = self.str2int[word]
        return self.emb_array[_int, :]

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def sentence2int(self, sentence, sentence_dict = False, add_eos = False, _class = 0):
        sentence = [word for word in wordpunct_tokenize(sentence)]
        try:
            pass
        except:
            print("FAILED to do wordpunct_tokenize with", sentence)
            TypeError("FAILED to do wordpunct_tokenize ")
        # print(sentence)
        sentence_in_int = []
        add_vector = []
        sentence_dictonary = {}
        i = 0
        for word in sentence:
            if self.check_for_uppers(word):
                add_vector.append((i, self.alphabet["<upper>"]))

            if self.check_in_dict(word):
                if self.lang == 'EN':
                    _emb = self.word_vectors[word]
                elif self.lang == 'NL':
                    _emb = self.str2emb(word)
                sentence_in_int.append(_emb)
                sentence_dictonary[i] = word
                i += 1

            else: #should be removed should always be incl.
                if not self.hasNumbers(word):
                    self.num_unknown += 1
                if word == "__eou__":
                    add_vector.append((i, self.alphabet["<eou>"]))
                    word = "<eou>"
                elif word in self.missing_words:
                    print(word in self.missing_words, word)
                    add_vector.append((i, self.alphabet[word]))
                else:

                    for letter in word:
                        if letter.lower() in self.alphabet:
                            add_vector.append(
                                (i, self.alphabet[letter.lower()]))
                        if letter.lower() in ".,\"\'!()*-":
                            add_vector.append((i, self.alphabet["<SYN>"]))
                        else:
                            add_vector.append((i, self.alphabet["<UNK>"]))

                sentence_dictonary[i] = word
                sentence_in_int.append(np.zeros((300,)))
                i += 1
                # i +=1
                # print("done with ",word,"now adding a space at ", i)

#                 sentence_in_int.append(np.zeros((300,)))
#                 add_vector.append((i,self.alphabet["<SPACE>"]))
#                 sentence_dictonary[i] = "<SPACE>"
#                 i += 1
        # print("seq_len", i)
        if add_eos:
            add_vector.append((i, self.alphabet["<eou>"]))
            sentence_dictonary[i] = word
            sentence_in_int.append(np.zeros((300,)))
            i += 1

        empty = np.zeros((i, self.extra_input))
        for j, letter in add_vector:
            empty[j, letter] = 1
        array = np.array(sentence_in_int)

        # if _class > 4 and self.num_unknown > 0:
        #     print("did not found", missing_words, "in", sentence)

        if sentence_dict:
            # print("seq_len", i)
            return array, empty, i, sentence_dictonary
        return array, empty, i

    def int2sentence(self, sentence_in_int, threshold = 0.05, seq_len = 100):
        sentence = []

        if len(sentence_in_int.shape) == 3:
            sentence_in_int = sentence_in_int[0, :, :]

        sentence_in_int = sentence_in_int[:seq_len, :]
        print("sentence2int shape", sentence_in_int.shape, seq_len)
        prev_was_letter = False
        word_of_letters = ""
        for i in range(seq_len):
            # print(np.max(sentence_in_int[i,300:]))
            # if i < 20:
                 # print(np.max(sentence_in_int[i,300:])<0.95,(1-threshold))

            if np.max(sentence_in_int[i, 300:-10]) < (1 - threshold):
                if prev_was_letter:
                    sentence.append(word_of_letters)
                    word_of_letters = ""
                start_time = time.time()
                word_dict = self.get_word(sentence_in_int[i, :300])
                print("getting ", word_dict[0][0],
                      " took: ", time.time() - start_time)
                if word_dict[0][1] < (1 - threshold):
                    print("wasnt sure about ",
                          word_dict[0][0], " other options ", word_dict)
                sentence.append(word_dict[0][0])
                prev_was_letter = False

            # not a word but an letter
            if np.max(sentence_in_int[i, 300:]) > (1 - threshold):
                letter = self.r_alphabet[np.argmax(sentence_in_int[i, 300:])]
                if letter in ["<UNK>", "<SYN>", '?']:
                    if prev_was_letter:
                        sentence.append(word_of_letters)
                        word_of_letters = ""
                        sentence.append(letter)
                        prev_was_letter = False
                    else:
                        sentence.append(letter)
                        prev_was_letter = False
                else:
                    word_of_letters += letter
                    prev_was_letter = True

        if prev_was_letter:
            sentence.append(word_of_letters)
        return sentence

    def get_name_certenty(self):

        text = self.total_conversation

        decision_vector = self.rnn_.get_name(
            conv_array = self.array, conv_ad_array=self.add_vector)

        return decision_vector

    def _to_max_words(self, array, add_vector, seq_len):
        if seq_len > 100:
            seq_len = 100
        x_array_ = np.zeros(
            (self.max_words, self.input_depth - self.extra_input))
        x_array_[:seq_len, :] = array[-self.max_words:, :]
        add_vector_ = np.zeros((self.max_words, self.extra_input))
        add_vector_[:seq_len, :] = add_vector[-self.max_words:, :]
        out_array = np.concatenate([x_array_, add_vector_], axis = 1)
        # print(out_array[-1,:])
        return out_array, seq_len

    def sentence2int_uppers(self, sentence_):
        add_vector = []
        i = 0
        last_spacebar = False
        for letter in sentence_:
            print(i, letter)
            if letter == " " and last_spacebar is False:
                i += 1
                last_spacebar = True
            else:
                last_spacebar = False
                if letter.isupper():
                    print(i, letter)
                    add_vector.append((i, self.alphabet["<upper>"]))
        return add_vector

    def int2sentence_uppers(self, sentence_in_int, sentence):
        add_vector = []
        for i in range(sentence_in_int.shape[0]):
            if sentence_in_int[i, 300 + self.alphabet["<upper>"]]:
                print(sentence[i])

    def check_for_uppers(self, word):
        for letter in word:
            if letter.isupper():
                return True
        return False


if __name__ == "__main__":
    # code to run a converation

    dialog_ = dialog(word2vec = True, lang = "NL")
    word_list = ["hello", 'hoi', "Hello", "HOI",
                 "LOL", 'Tako', "tako", "Tabak", "tabak", 'owahgo', '']
    for word in word_list:
        print(word, dialog_.check_in_dict(word), word in dialog_.str2int)

        #     _array, _add_array, _sequence_length_l = dialog_.sentence2int(
        #         'Hello my name is Tako')
        #     _array, seq_ = dialog_._to_max_words(
        #         _array, _add_array, _sequence_length_l)
        #     print(_array[:30, -5:])
        #     print("Initialized")
        # text = input()
        # if text == "\$end":
        #     break
        # dialog_.get_and_send_message(text)

        # i += 1
    print("Oke I hope to see again soon.")
