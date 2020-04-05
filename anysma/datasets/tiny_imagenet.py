"""Tiny ImageNet dataset for classification.

URL:
    https://tiny-imagenet.herokuapp.com/

LICENCE / TERMS / COPYRIGHT:
    -

Description:
    Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images.

Workaround copied from Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
import numpy as np


def load_data(path='tiny_imagenet.npz'):
    """Loads the Tiny ImageNet dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train, boxes_train), (x_val, y_val, boxes_val), (x_test)`.
    """
    path = get_file(path,
                    origin='http://tiny.cc/anysma_datasets_tinyimgnt',
                    file_hash='f6eb009c75776916595dc6c55974135f')
    f = np.load(path)
    x_train, y_train, boxes_train = f['train_images'], f['train_labels'], f['train_boxes']
    x_val, y_val, boxes_val = f['val_images'], f['val_labels'], f['val_boxes']
    x_test = f['test_images']
    f.close()
    return (x_train, y_train, boxes_train), (x_val, y_val, boxes_val), x_test
