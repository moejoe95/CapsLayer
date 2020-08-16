from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import capslayer as cl
import tensorflow as tf


def selfAttention(x, ch, name='attention'):
    '''
    Self-Attention mechanism from: https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    '''
    with tf.variable_scope(name):

        f = tf.layers.conv2d(x, ch // 8, kernel_size=1, strides=1) 
        g = tf.layers.conv2d(x, ch // 8, kernel_size=1, strides=1)
        h = tf.layers.conv2d(x, ch // 1, kernel_size=1, strides=1)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        shape = cl.shape(x)
        o = tf.reshape(o, shape=shape) # [bs, h, w, C]

        # check this out: SACN doesn't use this 1x1 conv, but SAGAN does
        o = tf.layers.conv2d(o, ch, kernel_size=1, strides=1)

        x = gamma * o + x

        return x


def hw_flatten(x) :
    shape = cl.shape(x)
    return tf.reshape(x, [-1, shape[1]*shape[2], shape[3]])
