import nltk
nltk.download('stopwords')
from nltk.tokenize import wordpunct_tokenize

def sentence2int(sentence, sentence_dict = False):
    try:
        sentence = [word for word in wordpunct_tokenize(sentence)]
    except:
        print("FAILED to do wordpunct_tokenize with", sentence)
        TypeError("FAILED to do wordpunct_tokenize ")

    sentence_in_int = []
    add_vector = []
    sentence_dictonary = {}
    i = 0
    for word in sentence:
        if check_for_uppers(word):
            add_vector.append((i, alphabet["<upper>"]))

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
