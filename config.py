import os

data_loc = "E:\\data\\MNIST"  # CHANGE this

config = {"MNIST": {"test" :
                        {"dir"   : os.path.join(data_loc, "test"),
                         "images": 't10k-images-idx3-ubyte',
                         "labels": 't10k-labels-idx1-ubyte',
                         "size"  : 10000},

                    "train":
                        {"dir"   : os.path.join(data_loc, "train"),
                         "images": 'train-images-idx3-ubyte',
                         "labels": 'train-labels-idx1-ubyte',
                         "size"  : 60000},
                    },

          }
PREPERATION_DIR = "./preperation_files/"
