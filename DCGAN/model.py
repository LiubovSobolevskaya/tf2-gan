"""Generator and Discriminator for Deep Convolutional Generative Adversarial Network"""

import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    """
    Generator produces an image from a random noise.
    
    Args:
       ngf (int): Number of filters.
    """

    def __init__(self, ngf, out_channels):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.main = tf.keras.Sequential([
            layers.Conv2DTranspose(
                self.ngf * 8,
                kernel_size=(4, 4),
                strides=1,
                padding="valid",
                use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(
                self.ngf * 4,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(
                self.ngf * 2,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(
                self.ngf,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(
                out_channels,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
        ])

    def call(self, input):
        return tf.keras.activations.tanh(self.main(input))


class Discriminator(tf.keras.Model):
    """
    Discriminator classifies the generated images as real or fake
    
    Args:
       ndf (int): Number of filters.
    """

    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.main = tf.keras.Sequential([
            layers.Conv2D(
                self.ndf,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                self.ndf * 2,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                self.ndf * 4,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                self.ndf * 8,
                kernel_size=(4, 4),
                strides=2,
                padding="same",
                use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=(4, 4), strides=1, use_bias=False),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, input):
        return self.main(input)
