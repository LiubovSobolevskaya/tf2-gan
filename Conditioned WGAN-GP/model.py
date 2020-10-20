"""Generator and Discriminator for Deep Convolutional Generative Adversarial Network"""

import tensorflow as tf
from tensorflow.keras import layers


class Film_Gen(tf.keras.Model):
    """
    produces parameter for Generator
    
    Args:
       ngf (int): Number of filters.
       out_channels (int): number of channels in the generated image
    """

    def __init__(self, ngf, output_channels):
        super(Film_Gen, self).__init__()
        self.ngf = ngf
        self.output_channels = output_channels
        self.linear_main1 = layers.Dense(self.ngf * 4)
        self.linear_main2 = layers.Dense(self.ngf * 2)
        self.linear_main3 = layers.Dense(self.ngf)

        near_zero_init = tf.keras.initializers.Orthogonal(1e-4)
        self.linear_mu1 = layers.Dense(
            self.ngf * 4, kernel_initializer=near_zero_init)
        self.linear_sigma1 = layers.Dense(
            self.ngf * 4, kernel_initializer=near_zero_init)

        self.linear_mu2 = layers.Dense(
            self.ngf * 2, kernel_initializer=near_zero_init)
        self.linear_sigma2 = layers.Dense(
            self.ngf * 2, kernel_initializer=near_zero_init)

        self.linear_mu3 = layers.Dense(
            self.ngf, kernel_initializer=near_zero_init)
        self.linear_sigma3 = layers.Dense(
            self.ngf, kernel_initializer=near_zero_init)

        self.act1 = layers.ReLU()
        self.act2 = layers.ReLU()
        self.act3 = layers.ReLU()

    @tf.function
    def call(self, inputs):
        out = self.linear_main1(inputs)
        out = self.act1(out)
        mu_1 = self.linear_mu1(out)
        log_gamma_1 = self.linear_sigma1(out)
        gamma_1 = tf.keras.activations.softplus(log_gamma_1)

        out = self.linear_main2(out)
        out = self.act2(out)
        mu_2 = self.linear_mu2(out)
        log_gamma_2 = self.linear_sigma2(out)
        gamma_2 = tf.keras.activations.softplus(log_gamma_2)

        out = self.linear_main3(out)
        out = self.act3(out)
        mu_3 = self.linear_mu3(out)
        log_gamma_3 = self.linear_sigma3(out)
        gamma_3 = tf.keras.activations.softplus(log_gamma_3)

        return mu_1, gamma_1, mu_2, gamma_2, mu_3, gamma_3


class Generator_FiLM(tf.keras.Model):
    """
    Generator produces an image from a random noise and condition vector.
    
    Args:
       ngf (int): Number of filters.
       out_channels (int): number of channels in the generated image
    """

    def __init__(self, ngf, image_size, output_channels):
        super(Generator_FiLM, self).__init__()
        self.output_channels = output_channels
        self.image_size = image_size
        self.ngf = ngf
        self.main = tf.keras.Sequential([
            layers.Dense(4 * 4 * 8 * self.ngf, use_bias=False),
            layers.ReLU(),
            layers.Reshape((4, 4, 8 * self.ngf))
        ])
        self.conv1 = layers.Conv2DTranspose(
            self.ngf * 4,
            kernel_size=(4, 4),
            strides=1,
            padding="same",
            use_bias=False)
        self.conv2 = layers.Conv2DTranspose(
            self.ngf * 2,
            kernel_size=(4, 4),
            strides=2,
            padding="same",
            use_bias=False)
        self.conv3 = layers.Conv2DTranspose(
            self.ngf,
            kernel_size=(4, 4),
            strides=2,
            padding="same",
            use_bias=False)
        self.conv4 = layers.Conv2DTranspose(
            self.output_channels,
            kernel_size=(4, 4),
            strides=2,
            padding="same")

        self.act1 = layers.ReLU()
        self.act2 = layers.ReLU()
        self.act3 = layers.ReLU()

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

        self.Film_Gen = Film_Gen(self.ngf, self.output_channels)
        self.flatten = layers.Flatten()

    @tf.function
    def call(self, inputs, training):
        img = inputs[0]
        condition = inputs[1]
        mu_1, gamma_1, mu_2, gamma_2, mu_3, gamma_3 = self.Film_Gen(condition)
        out = self.main(img)
        out = self.bn1(self.conv1(out), training=training)
        out = self.act1(out * gamma_1[:, tf.newaxis, tf.newaxis] +
                        mu_1[:, tf.newaxis, tf.newaxis])
        out = self.bn2(self.conv2(out), training=training)
        out = self.act2(out * gamma_2[:, tf.newaxis, tf.newaxis] +
                        mu_2[:, tf.newaxis, tf.newaxis])
        out = self.bn3(self.conv3(out), training=training)
        out = self.act3(out * gamma_3[:, tf.newaxis, tf.newaxis] +
                        mu_3[:, tf.newaxis, tf.newaxis])
        out = self.conv4(out)

        return tf.keras.activations.tanh(out)


class Generator(tf.keras.Model):
    """
    Generator produces an image from a random noise and condition vector.
    
    Args:
       ngf (int): Number of filters.
       out_channels (int): number of channels in the generated image.
    """

    def __init__(self, ngf, out_channels):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.main = tf.keras.Sequential([
            layers.Dense(4 * 4 * 8 * self.ngf, use_bias=False),
            layers.ReLU(),
            layers.Reshape((4, 4, 8 * self.ngf)),
            layers.Conv2DTranspose(
                self.ngf * 4,
                kernel_size=(4, 4),
                strides=1,
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
                out_channels, kernel_size=(4, 4), strides=2, padding="same"),
        ])

    @tf.function
    def call(self, inputs, training):
        out = tf.concat(values = [inputs[0], inputs[1]], axis=-1)
        return tf.keras.activations.tanh(self.main(out, training=training))


class Discriminator(tf.keras.Model):
    """
    Discriminator classifies the generated images as real or fake.
    
    Args:
       ndf (int): Number of filters.
       out_channels (int): number of channels in the generated image.
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
        return self.main(inputs, training=training)
