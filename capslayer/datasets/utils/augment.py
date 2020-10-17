
import tensorflow as tf
from tensorflow.keras import layers

class Augmenter:

    __init__(image_size=32, norm=1./255):
        self.resize = tf.keras.Sequential([
            preprocessing.Resizing(image_size, image_size),
            preprocessing.Rescaling(norm),
        ])

        self.randomFlipX = tf.keras.Sequential([
            preprocessing.RandomFlip("horizontal"),
        ])
        
        self.randomRotation = tf.keras.Sequential([
            preprocessing.RandomRotation(0.1),
        ])

        self.randomZoom = tf.keras.Sequential[
            preprocessing.RandomZoom(0.1),
        ])

        self.randomCrop = tf.keras.Sequential([
            preprocessing.RandomCrop(28, 28),
        ])

        self.centerCrop = tf.keras.Sequential([
            preprocessing.CenterCrop(28, 28),
        ])

        # additive gaussian noise
        gnoise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
        self.randomNoise = tf.keras.Sequential([
            tf.add(image, gnoise),
        ])


    def augment(image):
        image = self.resize(image)
        image = self.randomFlipX(image)
        image = self.randomZoom(image)
        return image
