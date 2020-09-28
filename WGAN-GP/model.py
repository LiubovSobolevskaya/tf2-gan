"""Generator and Discriminator for Deep Convolutional Generative Adversarial Network"""

import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    """
    Generator produces an image from a random noise.
    
    Args:
       ngf (int): Number of filters.
    """

    def __init__(self, ngf, latent_space, out_channels, output_dim):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.output_dim = output_dim
        self.main = tf.keras.Sequential([
            layers.Dense(
                4 * 4 * 8 * self.ngf,
                use_bias=False,
                input_shape=(latent_space, )),
            layers.ReLU(),
            layers.Reshape((4, 4, 8 * self.ngf)),
            layers.Conv2DTranspose(
                self.ngf * 4,
                kernel_size=(4, 4),
                strides=1,
                padding="valid",
                use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(
                self.ngf * 2,
                kernel_size=(4, 4),
                strides=2,
                padding="valid",
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

    def call(self, inputs, training):
        return tf.reshape(
            tf.keras.activations.tanh(self.main(inputs)), [-1, self.output_dim])


class Discriminator(tf.keras.Model):
    """
    Discriminator classifies the generated images as real or fake
    
    Args:
       ndf (int): Number of filters.
    """

    def __init__(self, ndf, image_size, output_channels):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.image_size = image_size
        self.output_channels = output_channels
        self.main = tf.keras.Sequential([
            layers.Conv2D(
                self.ndf,
                kernel_size=(5, 5),
                strides=2,
                padding="same",
                use_bias=False),           
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                self.ndf * 2,
                kernel_size=(5, 5),
                strides=2,
                padding="same",
                use_bias=False),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                self.ndf * 4,
                kernel_size=(5, 5),
                strides=2,
                padding="same",
                use_bias=False),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, inputs, training):
        output = tf.reshape(inputs, [-1, self.image_size , self.image_size , self.output_channels])
        return self.main(output)
