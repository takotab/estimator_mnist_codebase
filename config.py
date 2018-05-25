
import os
data_loc = "E://data//MNIST"

config = {"MNIST": {"test":
                    {"dir": os.path.join(data_loc, "test"),
                     "images": 't10k-images-idx3-ubyte',
                     "labels": 't10k-labels-idx1-ubyte'},

                    "train":
                        {"dir": os.path.join(data_loc, "train"),
                         "images": 'train-images-idx3-ubyte',
                         "labels": 'train-labels-idx1-ubyte'},
                    },


          "ink_file": {"dir": "E://data//enity_article_estimate//current_inky.ink",
                       "train": "E://data//enity_article_estimate//CoNLLt//current_inky.CoNLLt",
                       "val": "E://data//enity_article_estimate//CoNLLt//current_inky_val.CoNLLt",
                       "settings": {"entities": ['ORG', 'MISC', 'PER', 'LOC'],
                                    "intent": []}},

          "CoNLLt": {"word_sep": "\t",
                     "line_sep": ". Punc O"},


          "extra_info": {"alphabet": {"info": "alphabet",
                                      "meta": "deep"},
                         "street_part": {"info": ["straat", 'plein', 'weg', 'laan'],
                                         "meta": "both"},
                         "Wheather_words": {"info": ['weer', 'temperatuur', 'regen'],
                                            "meta": "sparse"},
                         "Salutation": {"info": ['naam', "heet", 'mr', 'mvr'],
                                        "meta": "sparse"},
                         }
          }
