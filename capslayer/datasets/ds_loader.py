import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader(object):
    """ Data Loader.
    """
    def __init__(self, 
                dataset,
                shape=[32, 32, 3],
                name="create_inputs"):
        self.dataset = dataset
        self.shape = shape
        self.name = name

        self.next_element = None
        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
        
        # check if dataset is available
        if dataset not in tfds.list_builders():
            raise NotImplementedError(dataset)
        else:
            print(dataset, 'dataset found in tfds ...')


    
    def __call__(self, batch_size, mode):
        with tf.name_scope(self.name):
            self.mode = mode.lower()
            self.batch_size = batch_size

            if self.mode == 'test':
                raise NotImplementedError('test')

            (self.ds_train, self.ds_test), self.ds_info = tfds.load(
                self.dataset,
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            iterator = self.get_iterator()

            if self.next_element is None:
                self.next_element = tf.compat.v1.data.Iterator.from_string_handle(self.handle, iterator.output_types).get_next()

            return iterator


    def get_iterator(self):

        dataset = self.ds_train if self.mode == 'train' else self.ds_test

        dataset = dataset.map(
            self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.mode == 'train':
            dataset = dataset.cache()
            dataset = dataset.shuffle(self.ds_info.splits['train'].num_examples)
            dataset = dataset.batch(self.batch_size)
        elif self.mode == 'eval':
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.cache()
            
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if self.mode == 'train':
            dataset = dataset.repeat()
            return tf.compat.v1.data.make_one_shot_iterator(dataset)
        else:
            dataset = dataset.repeat(1)
            return tf.compat.v1.data.make_initializable_iterator(dataset)


    def augment(self, image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (self.shape[0], self.shape[1]))
        image = tf.image.random_flip_left_right(image)
        if self.shape[2] == 3:
            image = tf.image.random_saturation(image, 0.6, 1.6)
            image = tf.image.random_brightness(image, 0.05)
            image = tf.image.random_contrast(image, 0.7, 1.3)

        return image, label