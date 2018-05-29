# code from https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.int32).newbyteorder('>')  # pylint : ignore
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            print("magic number is %i should be 2051" % (magic))
            # raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
            #    f.name))
        if rows != 28 or cols != 28:
            print('Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                  (f.name, rows, cols))
            # raise ValueError(
            #     'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
            #     (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            print("magic number is %i should be 2049" % (magic))
            # raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
            #                                                                f.name))


def _download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return filepath
    if not os.path.isdir(directory):
        os.mkdir(directory)
    url = 'http://yann.lecun.com/exdb/mnist/' + filename + '.gz'
    f, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    os.close(f)
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in:
        with open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        f_in.close()
    os.remove(zipped_filepath)
    return filepath

# https://pjreddie.com/projects/mnist-in-csv/


def mnist2csv(imgf, labelf, outf, n):
    if os.path.isfile(outf):
        if len((open(outf).readline()).split(",")) > 1:
            return outf
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for _ in range(n):
        image = [ord(l.read(1))]
        for __ in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

# made by me


import utils
CONFIG = utils.import_config()


def download():

    for test_train in CONFIG["MNIST"]:

        directory = CONFIG["MNIST"][test_train]["dir"]

        images_file = CONFIG["MNIST"][test_train]["images"]
        images_dir = _download(directory, images_file)
        # check_image_file_header(images_dir)

        labels_file = CONFIG["MNIST"][test_train]["labels"]
        labels_dir = _download(directory, labels_file)
        # check_labels_file_header(labels_dir)

        CONFIG["MNIST"][test_train]["labels_dir"] = labels_dir
        CONFIG["MNIST"][test_train]["images_dir"] = images_dir

        csv_dir = os.path.join(
            directory, "mnist_"+test_train+".csv")
        CONFIG["MNIST"][test_train]["csv"] = csv_dir

        # makes the csv file of the data.
        mnist2csv(images_dir, labels_dir, csv_dir,
                  CONFIG["MNIST"][test_train]["size"])
        assert len((open(csv_dir).readline()).split(",")
                   ) > 1, "The MNIST " + test_train + " dataset is not the right size"


if __name__ == "__main__":
    print(download())
