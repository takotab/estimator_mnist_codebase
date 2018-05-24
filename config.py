alphabet = [abc for abc in 'abcdefghijklmnopqrstuvwxyz0123456789?']
alphabet.append("<UNK>")
alphabet.append("<SYN>")
alphabet.append("<upper>")
alphabet.append("<eou>")

prep_file_dir = "E://data//enity_article_estimate//preperation_files"

config = {"CoNLL": {"train": "E://data//enity_article_estimate//CoNLLt//ned.train.CoNLLt",
                    "val": "E://data//enity_article_estimate//CoNLLt//ned.test.CoNLLt",
                    "settings": None},

          "ink_file": {"dir": "E://data//enity_article_estimate//current_inky.ink",
                       "train": "E://data//enity_article_estimate//CoNLLt//current_inky.CoNLLt",
                       "val": "E://data//enity_article_estimate//CoNLLt//current_inky_val.CoNLLt",
                       "settings": {"entities": ['ORG', 'MISC', 'PER', 'LOC'],
                                    "intent": []}},

          "CoNLLt": {"word_sep": "\t",
                     "line_sep": ". Punc O"},

          "Preparation_files": {"first_name": prep_file_dir + "//list_w_first_names.txt",
                                "last_name": prep_file_dir + "//list_w_last_names.txt",
                                "cities": prep_file_dir + "//list_w_cities.txt",
                                "embedding": prep_file_dir + "//nl-embedding.pckl",
                                "lists": prep_file_dir + "//lists.pkl",
                                "pointers": prep_file_dir + "//pointers.pkl"},

          "extra_info": {"alphabet": {"info": alphabet,
                                      "meta": "deep"},
                         "street_part": {"info": ["straat", 'plein', 'weg', 'laan'],
                                         "meta": "both"},
                         "Wheather_words": {"info": ['weer', 'temperatuur', 'regen'],
                                            "meta": "sparse"},
                         "Salutation": {"info": ['naam', "heet", 'mr', 'mvr'],
                                        "meta": "sparse"},
                         }
          }
