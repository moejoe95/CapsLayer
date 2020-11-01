# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

"""Input utility functions for reading Cifar10 dataset.

Handles reading from Cifar10 dataset saved in binary original format. Scales and
normalizes the images as the preprocessing step. It can distort the images by
random cropping and contrast adjusting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from capslayer.data.utils.download_utils import maybe_download_and_extract
from capslayer.data.datasets.cifar10.writer import tfrecord_runner
from capslayer.data.utils.augment import Augmenter


def parse_fun(serialized_example):
    """ Data parsing function.
    """
    features = tf.io.parse_single_example(serialized_example,
                                       features={'image': tf.io.FixedLenFeature([], tf.string),
                                                 'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image'], tf.float32)

    image = augmenter.augment(image)
    
    label = tf.cast(features['label'], tf.int32)
    features = {'images': image, 'labels': label}
    return(features)


class DataLoader(object):
    """ Data Loader.
    """
    def __init__(self, path=None,
                 num_works=1,
                 splitting="TVT",
                 one_hot=False,
                 name="create_inputs"):

        if path is None:
            path = os.path.join(os.environ["HOME"], ".cache", "capslayer", "datasets", "cifar10")
            os.makedirs(path, exist_ok=True)
        elif os.access(path, os.F_OK):
            path = path if os.path.basename(path) == "cifar10" else os.path.join(path, "cifar10")
            os.makedirs(path, exist_ok=True)
        elif os.access(path, os.W_OK):
            raise IOError("Permission denied! Path %s is not writable." % (str(path)))

        # data downloaded and data extracted?
        maybe_download_and_extract("cifar10", path)
        # data tfrecorded?
        tfrecord_runner(path, force=False)

        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
        self.next_element = None
        self.path = path
        self.name = name
        self.augment = Augment(32)


    def __call__(self, batch_size, mode):
        """
        Args:
            batch_size: Integer.
            mode: Running phase, one of "train", "test" or "eval"(only if splitting='TVT').
        """
        with tf.name_scope(self.name):
            mode = mode.lower()
            modes = ["train", "test", "eval"]
            filenames = [os.path.join(self.path, '%s_cifar10.tfrecord' % mode)]

            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parse_fun)
            dataset = dataset.batch(batch_size)

            if mode == "train":
                dataset = dataset.shuffle(buffer_size=50000)
                dataset = dataset.repeat()
                iterator = dataset.make_one_shot_iterator()
            elif mode == "eval":
                dataset = dataset.repeat(1)
                iterator = dataset.make_initializable_iterator()
            elif mode == "test":
                dataset = dataset.repeat(1)
                iterator = dataset.make_one_shot_iterator()

            if self.next_element is None:
                self.next_element = tf.compat.v1.data.Iterator.from_string_handle(self.handle, iterator.output_types).get_next()

            return (iterator)