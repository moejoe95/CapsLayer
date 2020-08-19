import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

import tensorflow_datasets as tfds

DS_NAME = 'imagenette/160px-v2'
DS_PATH = DS_NAME + '/0.1.0/'

def parse_fun(serialized_example):
    """ Data parsing function.
    """

    features = tf.parse_single_example(serialized_example,
                                       features={'image': tf.FixedLenFeature([], tf.string),
                                                 'label': tf.FixedLenFeature([], tf.int64)}
                                        )

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, shape=[160 * 160 * 3])
    image.set_shape([160 * 160 * 3])
    image = tf.cast(image, tf.float32) / 255. # * (2. / 255) - 1.

    label = tf.cast(features['label'], tf.int64)

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

        base_path = path
        path = path if os.path.basename(path) == DS_NAME else os.path.join(base_path, DS_NAME)
        os.makedirs(path, exist_ok=True)

        # data downloaded and data extracted?
        ds = tfds.load(DS_NAME, split='train', shuffle_files=True, data_dir=base_path)

        ds_file_path = os.path.join(base_path, DS_PATH)
        for filename in os.listdir(ds_file_path):
            if 'train' in filename:
                os.rename(ds_file_path + filename, ds_file_path + '../train_imagenette.tfrecord')           
            elif 'validation' in filename:
                os.rename(ds_file_path + filename, ds_file_path + '../eval_imagenette.tfrecord')  

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
            filenames = [os.path.join(self.path, '%s_imagenette.tfrecord' % mode)]
            print('mode:', mode)
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parse_fun)
            dataset = dataset.batch(batch_size)

            if mode == "train":
                dataset = dataset.shuffle(buffer_size=30000)
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