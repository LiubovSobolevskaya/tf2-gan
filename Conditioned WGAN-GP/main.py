"""Main file for training of Deep Convolutional Generative Adversarial Network to generate images using condition."""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from model import Discriminator, Generator_FiLM, Generator

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 1e-4, 'learning rate')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('epochs', 50, 'epochs')
flags.DEFINE_integer('disc_iters', 5, 'number of critic iters per gen iter')
flags.DEFINE_integer('num_examples', 16, 'number of examples to generate')
flags.DEFINE_integer('num_filters', 64, 'number of filters')
flags.DEFINE_integer('image_size', 32, 'size of image')
flags.DEFINE_integer('lambda', 10, 'Gradient penalty lambda hyperparameter')
flags.DEFINE_integer('latent_vector', 100, 'size of latent space')
flags.DEFINE_string('dataset', 'mnist', 'mnist')
flags.DEFINE_string('save_folder', 'images',
                    'folder to save generated images')
flags.DEFINE_boolean('use_FiLM', True, 'use FiLM layers')


def main(_):

    strategy = tf.distribute.MirroredStrategy()

    NUM_GPU = len(tf.config.experimental.list_physical_devices('GPU'))

    train_ds, ds_info = tfds.load(
        FLAGS.dataset, split='train', shuffle_files=True, with_info=True)

    if FLAGS.dataset == 'mnist':
        output_channels = 1
    else:    
        output_channels = 3
        
    num_classes = ds_info.features['label'].num_classes

    OUTPUT_DIM = FLAGS.image_size * FLAGS.image_size * (
        output_channels + num_classes)

    def label_to_channels(labels, shape):
        labels = tf.one_hot(labels, num_classes)
        labels = tf.reshape(labels, shape=[shape, 1, 1, num_classes])
        labels = tf.tile(labels, [1, FLAGS.image_size, FLAGS.image_size, 1])
        return labels

    def preprocess(image):
        """Normalize the images to [-1.0, 1.0]"""
        label = image['label']
        image = image['image']
        image = tf.image.resize_with_pad(image, FLAGS.image_size,
                                         FLAGS.image_size)
        return (tf.cast(image, tf.float32) - 127.5) / 127.5, label

    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
    train_ds = train_ds.batch(FLAGS.batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    with strategy.scope():

        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.lr, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.lr, beta_1=0.5, beta_2=0.9)

        input1 = tf.keras.Input(
            shape=(FLAGS.latent_vector), name="latent_vector")
        input2 = tf.keras.Input(shape=(num_classes, ), name="condition")
        if FLAGS.use_FiLM:
            outputs = Generator_FiLM(FLAGS.num_filters, FLAGS.image_size,
                                     output_channels)([input1, input2])
        else:
            outputs = Generator(FLAGS.num_filters,
                                output_channels)([input1, input2])
        generator = tf.keras.Model(inputs=[input1, input2], outputs=outputs)

        inputs = tf.keras.Input(
            shape=(FLAGS.image_size, FLAGS.image_size,
                   output_channels + num_classes),
            name="imgs")
        outputs = Discriminator(FLAGS.num_filters, FLAGS.image_size,
                                output_channels + num_classes)(inputs)
        discriminator = tf.keras.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def train_gen():

        noise = tf.random.normal([FLAGS.batch_size // NUM_GPU, FLAGS.latent_vector])
        generated_labels = tf.random.uniform(
            (FLAGS.batch_size // NUM_GPU, ),
            maxval=num_classes,
            dtype=tf.dtypes.int32)
        gen_labels = tf.one_hot(generated_labels, num_classes)
        generated_labels = label_to_channels(generated_labels,
                                             FLAGS.batch_size // NUM_GPU)
        with tf.GradientTape() as gen_tape:

            generated_images = generator((noise, gen_labels), training=True)

            fake_output = discriminator(
                tf.concat([generated_images, generated_labels], axis=-1),
                training=False)
            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))

        return tf.reduce_mean(gen_loss)

    @tf.function
    def train_disc(images, labels):
        labels = label_to_channels(labels, images.shape[0])
        images = tf.concat([images, labels], axis=-1)

        noise = tf.random.normal([images.shape[0], FLAGS.latent_vector])
        generated_labels = tf.random.uniform(
            (images.shape[0], ), maxval=num_classes, dtype=tf.dtypes.int32)
        gen_labels = tf.one_hot(generated_labels, num_classes)

        generated_labels = label_to_channels(generated_labels, images.shape[0])

        fake_images = generator((noise, gen_labels), training=True)
        fake_images = tf.concat(values = [fake_images, generated_labels], axis=-1)

        with tf.GradientTape() as disc_tape:

            disc_real = discriminator(images, training=True)
            disc_fake = discriminator(fake_images, training=True)
            disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random.uniform(
                shape=[
                    images.shape[0],
                    1,
                ], maxval=1.)

            differences = tf.reshape(fake_images,
                                     [-1, OUTPUT_DIM]) - tf.reshape(
                                         images, [-1, OUTPUT_DIM])

            interpolates = tf.reshape(
                images, [-1, OUTPUT_DIM]) + (alpha * differences)
            gradients = tf.gradients(
                discriminator(
                    tf.reshape(interpolates, [
                        -1, FLAGS.image_size, FLAGS.image_size,
                        output_channels + num_classes
                    ])), [interpolates])[0]

            slopes = tf.math.sqrt(
                tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

            disc_loss += 10 * gradient_penalty

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))
        return tf.reduce_mean(disc_loss)

    @tf.function
    def distributed_disc_step(dist_inputs, dist_labels):
        per_replica_disc_loss = strategy.run(
            train_disc, args=[dist_inputs, dist_labels])
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_disc_loss, axis=None)

    @tf.function
    def distributed_gen_step():
        per_replica_gen_loss = strategy.run(train_gen, args=())
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_gen_loss, axis=None)

    def save_images(model, ep, vector, noise_labels):

        predictions = tf.clip_by_value(
            model((vector, noise_labels), training=False), -1, 1)
        plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            pred = tf.reshape(
                predictions[i],
                [FLAGS.image_size, FLAGS.image_size, output_channels])
            plt.imshow((pred.numpy() * 127.5 + 127.5).astype(np.uint8))
            plt.axis('off')

        plt.savefig(
            FLAGS.save_folder + '/image_at_epoch_{:02d}.png'.format(ep))

    if not os.path.exists(FLAGS.save_folder):
        os.makedirs(FLAGS.save_folder)

    noise_vector = tf.random.normal([FLAGS.num_examples, FLAGS.latent_vector])
    noise_labels = np.arange(16) % num_classes
    noise_labels = tf.one_hot(noise_labels, num_classes)

    for epoch in tqdm(range(FLAGS.epochs)):

        iterator = iter(train_ds)
        gen_loss = 0
        disc_loss = 0
        num_batch = 0
        iterations = 0
        flag = True
        while flag:
            gen_loss += distributed_gen_step()
            iterations += 1
            for _ in range(FLAGS.disc_iters):
                optional = iterator.get_next_as_optional()
                if not optional.has_value().numpy():
                    flag = False
                else:
                    data, label = optional.get_value()
                    d_loss = distributed_disc_step(data, label)
                    disc_loss += d_loss
                    num_batch += 1

        disc_loss /= num_batch
        gen_loss /= iterations
        print("Epoch {}, gen_loss  {:.5f} \n disc_loss {:.5f}\n".format(
            epoch, gen_loss, disc_loss))

        save_images(generator, epoch, noise_vector, noise_labels)

    save_images(generator, FLAGS.epochs, noise_vector, noise_labels)
     
        
    anim_file = 'dcgan_cifar10.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(FLAGS.save_folder + '/image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    embed.embed_file(anim_file)
    

if __name__ == '__main__':
    app.run(main)
