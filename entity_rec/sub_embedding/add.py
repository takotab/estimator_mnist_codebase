import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.tokenize import wordpunct_tokenize

def add_subword(sentence,subword_dict = None, additional_dict = None, dropout = 1):
    """
    this will make an n-gram of the words in the sentence.
    you can easily add stuff by using additional_dict.
    Any if the word contians the key the value (must be int) will be +1. 

    dropout is 1 by default if your training you can change this to accomodate regularization
    the value of dropout is the percentage to keep

    """
    if subword_dict is None:
        subword_dict = {abc: i for i, abc in enumerate(
            'abcdefghijklmnopqrstuvwxyz0123456789?')}
    
    if additional_dict is None:
        additional_dict = {}
        # examples
        additional_dict["en"] = 38
        additional_dict["sch"] = 39

    for key in additional_dict:
        value = additional_dict[key]
        assert type(value) == int
        subword_dict[key] = value

    max_value = find_max_sub_dict_value(subword_dict)

    sentence = [word for word in wordpunct_tokenize(sentence)]
    subword = np.zeros((len(sentence),max_value))

    for word_num, word in enumerate(sentence):
        for key in subword_dict:
            if word.__contains__(key):
                if np.random.rand() > dropout:
                    subword[word_num, subword_dict[key]] += 1
    
    return subword

def add_subsentence(sentence,subsentence_dict = None, dropout = 1):
    """
    NOTE: this makes use of regex

    this will make an n-gram of the sentence.
    you can easily add stuff by using additional_dict.
    Any if the word matches the regex the key the value (must be int) will be +1. 

    dropout is 1 by default if your training you can change this to accommodate regularization
    the value of dropout is the percentage to keep

    """
    if subsentence_dict is None:
        subsentence_dict = {r"(\d+)": 1,
                           "[0-9][0-9][0-9][0-9][A-Z][A-Z]": 2,
                           "([A-z]*)(laan|hof|straat|pad|markt|kade|boulevard|tuin|hof|steeg])( )([0-z]*)":3,
                           "(thuis.*vesta.*pro)": 4,
                           "(vesta.*pro.*thuis)": 4,
                           "(datafreeze)": 5,
                           "(recuperatieverlof.*glijsaldo.*negatief)": 6,
                           "(recuperatieverlof.*negatief.*glijsaldo)": 6,
                           "(negatief.*glijsaldo.*recuperatieverlof)": 6,
                           "(glijsaldo.*negatief.*recuperatieverlof)": 6,
                           "(tikkingen)": 7,
                           "(inhaalrust)": 7,
                           "verlofaanvraag.*(Outlook)": 8,
                           r"verlofaanvraag.*([e\s]mail)": 8,
                           "(Outlook).*verlofaanvraag": 8,
                           r"([e\s]mail).*verlofaanvraag": 8,
                           "(verlof).*(uren)": 9,
                           "(verlof).*(dag)": 9,
                           "(uren).*(verlof)": 9,
                           "(dag).*(verlof)": 9,
                           }
    
    max_value = find_max_sub_dict_value(subsentence_dict)

    subsentence = np.zeros((max_value))

    for regex in subsentence_dict:
        entity = re.findall(regex, sentence)
        if len(entity) and np.random.rand() > dropout:
            subsentence[subsentence_dict[regex]] += 1

    return subsentence


def find_max_sub_dict_value(sub_dict):
    max_value = 0
    for key in sub_dict:
        if sub_dict[key] > max_value:
            max_value = sub_dict[key]
    return max_value