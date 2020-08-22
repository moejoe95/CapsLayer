from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import cfg

def residualConvs(inputs, 
                conv_params,
                name="res_conv_layers"):

    # conv1
    conv1 = tf.layers.conv2d(inputs,
                            **conv_params,
                            padding='SAME',
                            name="res_conv1_layer")
    conv1_batched = tf.keras.layers.BatchNormalization()(conv1, training=cfg.is_training)
    conv1_act = tf.nn.relu(conv1_batched)

    # conv2
    conv2 = tf.layers.conv2d(conv1_act,
                            **conv_params,
                            padding='SAME',
                            name="res_conv2_layer")
    conv2_batched = tf.keras.layers.BatchNormalization()(conv2, training=cfg.is_training)
    conv2_act = tf.nn.relu(conv2_batched)

    # conv3
    conv3 = tf.layers.conv2d(conv2_act,
                            **conv_params,
                            padding='SAME',
                            name="res_conv3_layer")
    conv3_batched = tf.keras.layers.BatchNormalization()(conv3, training=cfg.is_training)
    
    # first residual connection
    conv3 = tf.keras.layers.Add()([conv1_batched, conv3_batched])
    conv3 = tf.nn.relu(conv3)

    # conv4
    conv4 = tf.layers.conv2d(conv3,
                            **conv_params,
                            padding='SAME',
                            name="res_conv4_layer")
    conv4_batched = tf.keras.layers.BatchNormalization()(conv4, training=cfg.is_training)
    conv4_act = tf.nn.relu(conv4_batched)

    # conv5
    conv5 = tf.layers.conv2d(conv4_act,
                            **conv_params,
                            padding='SAME',
                            name="res_conv5_layer")
    conv5_batched = tf.keras.layers.BatchNormalization()(conv5, training=cfg.is_training)
    conv5_act = tf.nn.relu(conv5_batched)

    # conv6
    conv6 = tf.layers.conv2d(conv5_act,
                            **conv_params,
                            padding='VALID',
                            name="res_conv6_layer")
    conv6_batched = tf.keras.layers.BatchNormalization()(conv6, training=cfg.is_training)

    # resize conv4 to size of conv6 by convolution
    conv4 = tf.layers.conv2d(conv4_batched,
                            **conv_params,
                            padding='VALID',
                            name="res_conv7_layer")

    # second residal connection
    conv6 = tf.keras.layers.Add()([conv4, conv6_batched])
    return tf.nn.relu(conv6)                               
        