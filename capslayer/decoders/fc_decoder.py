import tensorflow as tf
import numpy as np

class FCDecoderNet(object):

    def __init__(self, height, width, channels, num_label, labels, poses):
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label
        self.labels = labels
        self.poses = poses


    def reconstruct_image(self):
        '''
        Reconstruction network from CapsLayer, needs ~1.4m parameters for 28x28x1 images!
        '''
        with tf.compat.v1.variable_scope('Decoder'):
            labels = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
            labels_one_hoted = tf.reshape(labels, (-1, self.num_label, 1, 1))

            masked_caps = tf.multiply(self.poses, labels_one_hoted)
            num_inputs = np.prod(masked_caps.get_shape().as_list()[1:])
            active_caps = tf.reshape(masked_caps, shape=(-1, num_inputs))

            fc1 = tf.layers.dense(active_caps, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)

            num_outputs = self.height * self.width * self.channels
            recon_imgs = tf.layers.dense(fc2,
                                              units=num_outputs,
                                              activation=tf.sigmoid)

            return recon_imgs, labels_one_hoted


    def reconstruct_image_v2(self):
        '''
        Reconstruction network from gamma-capsule-network, needs ~1.4m parameters for 28x28x1 images!
        '''
        with tf.compat.v1.variable_scope('Decoder'):
            labels = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
            labels_one_hoted = tf.reshape(labels, (-1, self.num_label, 1, 1))

            num_outputs = self.height * self.width * self.channels

            flatten = tf.keras.layers.Flatten()
            fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
            fc2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
            fc3 = tf.keras.layers.Dense(num_outputs, activation=tf.sigmoid)

            x = flatten(self.poses)
            x = fc1(x)
            x = fc2(x)
            recon_imgs = fc3(x)

            return recon_imgs, labels_one_hoted
