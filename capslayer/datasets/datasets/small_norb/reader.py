"""Input utility functions for reading small-norb dataset.

Handles reading from small-norb dataset saved in binary original format. Scales and
normalizes the images as the preprocessing step. It can distort the images by
random cropping and contrast adjusting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from capslayer.data.utils.download_utils import maybe_download_and_extract
from capslayer.data.datasets.small_norb.writer import tfrecord_runner


def parse_fun(serialized_example):
    """ Data parsing function.
    """
    features = tf.parse_single_example(serialized_example,
                                       features={'image': tf.FixedLenFeature([], tf.string),
                                                 'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, shape=[32 * 32 * 3])
    image.set_shape([32 * 32 * 3])
    image = tf.cast(image, tf.float32) / 255. # * (2. / 255) - 1.
    label = tf.cast(features['label'], tf.int32)
    features = {'images': image, 'labels': label}
    return(features)


class DataLoader(object):
    """ Data Loader.
    """
    def __init__(self, path,
                 num_works=1,
                 splitting="TVT",
                 one_hot=False,
                 name="create_inputs"):

        path = path if os.path.basename(path) == "small_norb" else os.path.join(path, "small_norb")
        os.makedirs(path, exist_ok=True)

        # data downloaded and data extracted?
        maybe_download_and_extract("small_norb", path)
        # data tfrecorded?
        tfrecord_runner(path, force=False)

        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = None
        self.path = path
        self.name = name

    def __call__(self, batch_size, mode):
        """
        Args:
            batch_size: Integer.
            mode: Running phase, one of "train", "test" or "eval"(only if splitting='TVT').
        """
        with tf.name_scope(self.name):
            mode = mode.lower()
            modes = ["train", "test", "eval"]
            filenames = [os.path.join(self.path, '%s_small_norb.tfrecord' % mode)]

            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parse_fun)
            dataset = dataset.batch(batch_size)

            if mode == "train":
                # TODO check buffer size
                dataset = dataset.shuffle(buffer_size=2000)
                dataset = dataset.repeat()
                iterator = dataset.make_one_shot_iterator()
            elif mode == "eval":
                dataset = dataset.repeat(1)
                iterator = dataset.make_initializable_iterator()
            elif mode == "test":
                dataset = dataset.repeat(1)
                iterator = dataset.make_one_shot_iterator()

            if self.next_element is None:
                self.next_element = tf.data.Iterator.from_string_handle(self.handle, iterator.output_types).get_next()

            return(iterator)
