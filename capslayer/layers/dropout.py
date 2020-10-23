import tensorflow as tf
import capslayer as cl

seed = 20201022

def dropout(pose, drop_ratio=0.5, drop_mode='NEURON', name='dropout_layer'):
    '''Capsule dropout layer.

    Args:
        pose: 7-D tensor representing capsule poses.
        drop_ratio: 
        drop_mode: either NEURON or VECTOR. With NEURON each neuron is set 
        to 0 with a probability of drop_ratio independently. VECTOR means
        that whole capsules are dropped.
        name: name of dropout layer

    Returns:
        pose: 7-D tensor representing capsule poses
        activation: 5-D tensor representing capsule activations

    '''
    mask = None

    if drop_mode == 'VECTOR':
        mask = cl.shape(pose)
        mask[-2] = 1

    pose = tf.keras.layers.Dropout(drop_ratio, 
                                noise_shape=mask, 
                                seed=seed, 
                                name=name) (pose)

    activation = cl.norm(pose, axis=(-2, -1))
    activation = tf.clip_by_value(activation, 1e-20, 1. - 1e-20)

    return pose, activation
