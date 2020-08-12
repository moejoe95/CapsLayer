import tensorflow as tf
import numpy as np

class DeconvDecoderNet(object):

    def __init__(self, height, width, channels, num_label, labels, probs, vec_shape):
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label
        self.labels = labels
        self.probs = probs
        self.vec_shape = vec_shape


    def mask(self):
        '''
        @ Aryan Mobiny https://github.com/amobiny/Deep_Capsule_Network/blob/master/models/base_model.py
        '''
        with tf.variable_scope('Masking'):
            labels = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
            self.labels_one_hoted = tf.reshape(labels, (-1, self.num_label, 1, 1))
            # [?, 10] (one-hot-encoded predicted labels)

            # TODO do not hardcode this
            self.is_training = tf.constant(True)

            reconst_targets = tf.cond(self.is_training,  # condition
                                      lambda: tf.cast(self.labels, dtype='float32'),  # if True (Training)
                                      lambda: self.labels_one_hoted,  # if False (Test)
                                      name="reconstruction_targets")
            # [?, 10]
            self.output_masked = tf.multiply(self.probs, tf.expand_dims(reconst_targets, -1))
            # [?, 2, 16]


    def reconstruct_image(self):
        '''
        @ Aryan Mobiny https://github.com/amobiny/Deep_Capsule_Network/blob/master/models/Deep_CapsNet.py
        
        needs only ~4K parameters!
        '''
        self.mask()
        with tf.variable_scope('Deconv_Decoder'):
            cube_size = np.sqrt(self.vec_shape[0]).astype(int)
            decoder_input = tf.reshape(self.output_masked, [-1, self.num_label, cube_size, cube_size])
            cube = tf.transpose(decoder_input, [0, 2, 3, 1])
            print('cube', cube.shape)

            conv_rec1_params = {"filters": 8,
                                "kernel_size": 2,
                                "strides": 1,
                                "padding": "same",
                                "activation": tf.nn.relu}

            conv_rec2_params = {"filters": 16,
                                "kernel_size": 3,
                                "strides": 1,
                                "padding": "same",
                                "activation": tf.nn.relu}

            conv_rec3_params = {"filters": 16,
                                "kernel_size": 3,
                                "strides": 1,
                                "padding": "same",
                                "activation": tf.nn.relu}

            conv_rec4_params = {"filters": self.channels,
                                "kernel_size": 3,
                                "strides": 1,
                                "padding": "same",
                                "activation": None}

            conv_rec1 = tf.layers.conv2d(cube, name="conv1_rec", **conv_rec1_params)
            res1 = tf.image.resize_images(conv_rec1, (8, 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv_rec2 = tf.layers.conv2d(res1, name="conv2_rec", **conv_rec2_params)
            res2 = tf.image.resize_images(conv_rec2, (17, 17), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv_rec3 = tf.layers.conv2d(res2, name="conv3_rec", **conv_rec3_params)
            res3 = tf.image.resize_images(conv_rec3, (self.height*2, self.width*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            recon_imgs = tf.layers.conv2d(res3, name="conv4_rec", **conv_rec4_params)

            return tf.reshape(recon_imgs, shape=(-1, self.height * self.width * self.channels)), self.labels_one_hoted
