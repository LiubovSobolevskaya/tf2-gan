"""Main file for training of Deep Convolutional Generative Adversarial Network to generate images"""
import glob
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_docs.vis.embed as embed
from absl import app, flags
from tqdm import tqdm

from model import Discriminator, Generator

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 1e-4, 'learning rate')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('epochs', 50, 'epochs')
flags.DEFINE_integer('disc_iters', 5, 'number of critic iters per gen iter')
flags.DEFINE_integer('num_examples', 16, 'number of examples to generate')
flags.DEFINE_integer('num_filters', 64, 'number of filters')
flags.DEFINE_integer('image_size', 64, 'size of image')
flags.DEFINE_integer('lambda', 10, 'Gradient penalty lambda hyperparameter')
flags.DEFINE_integer('latent_vector', 100, 'size of latent space')
flags.DEFINE_string('dataset', 'cifar10', 'lsun/bedroom|cifar10|mnist')
flags.DEFINE_string('save_folder', 'images_cifar', 'folder tosave generated images')

def main(_):

    strategy = tf.distribute.MirroredStrategy()

    train_ds, ds_info = tfds.load(
        FLAGS.dataset, split='train', shuffle_files=True, with_info=True)
    
    #dataset is very big, don't want to wait long
    if FLAGS.dataset == 'lsun/bedroom':
        train_ds = train_ds.take(300000)
        output_channels = 3
    
    if FLAGS.dataset == 'cifar10':
        output_channels = 3 
    if FLAGS.dataset == 'mnist':  
        output_channels = 1

    OUTPUT_DIM = FLAGS.image_size * FLAGS.image_size * output_channels

    def preprocess(image):
        """Normalize the images to [-1.0, 1.0]"""
        image = image['image']
        image = tf.image.resize_with_pad(image, FLAGS.image_size,
                                         FLAGS.image_size)

        return (tf.cast(image, tf.float32) - 127.5) / 127.5

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

        inputs = tf.keras.Input(
            shape=(FLAGS.latent_vector, ), name="latent_vector")
        outputs = Generator(FLAGS.num_filters, FLAGS.latent_vector,
                            output_channels, OUTPUT_DIM)(inputs)
        generator = tf.keras.Model(inputs=inputs, outputs=outputs)

        inputs = tf.keras.Input(
            shape=(
                FLAGS.image_size * FLAGS.image_size * output_channels),
            name="imgs")
        outputs = Discriminator(FLAGS.num_filters, FLAGS.image_size, output_channels)(inputs)
        discriminator = tf.keras.Model(inputs=inputs, outputs=outputs)

    seed = tf.random.normal([FLAGS.num_examples, FLAGS.latent_vector])

    @tf.function
    def train_gen():

        noise = tf.random.normal([FLAGS.batch_size // 4, FLAGS.latent_vector])

        with tf.GradientTape() as gen_tape:

            generated_images = generator(noise, training=True)
            fake_output = discriminator(
                tf.reshape(generated_images, [-1, OUTPUT_DIM]), training=False)
            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))

        return tf.reduce_mean(gen_loss)

    checkpoint_dir = 'training_checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

    @tf.function
    def train_disc(images):
        image = tf.reshape(images, [-1, OUTPUT_DIM])
        noise = tf.random.normal([images.shape[0], FLAGS.latent_vector])
        fake_images = generator(noise, training=True)
        with tf.GradientTape() as disc_tape:

            disc_real = discriminator(images, training=True)
            disc_fake = discriminator(fake_images, training=True)
            disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random.uniform(
                shape=[
                    images.shape[0],
                    1,
                ], minval=0., maxval=1.)

            differences = fake_images - image

            interpolates = image + (alpha * differences)
            gradients = tf.gradients(
                discriminator(interpolates), [interpolates])[0]

            slopes = tf.math.sqrt(
                tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

            disc_loss += 10 * gradient_penalty

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))
        return tf.reduce_mean(disc_loss)

    checkpoint_dir = 'training_checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

    @tf.function
    def distributed_disc_step(dist_inputs):
        per_replica_disc_loss = strategy.run(train_disc, args=[dist_inputs])
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_disc_loss, axis=None)

    @tf.function
    def distributed_gen_step():
        per_replica_gen_loss = strategy.run(train_gen, args=())
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_gen_loss, axis=None)

    def generate_and_save_images(model, ep, vector):

        predictions = tf.clip_by_value(model(vector, training=False), -1, 1)
        plt.figure(figsize=(5, 5))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            pred = tf.reshape(
                predictions[i],
                [FLAGS.image_size, FLAGS.image_size, output_channels])
            plt.imshow((pred.numpy() * 127.5 + 127.5).astype(np.uint8))
            plt.axis('off')

        plt.savefig(FLAGS.save_folder +'/image_at_epoch_{:02d}.png'.format(ep))
    
    if not os.path.exists(FLAGS.save_folder):
         os.makedirs(FLAGS.save_folder)
    
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
                if optional.has_value().numpy() == False:
                    flag = False
                else:
                    data = optional.get_value()
                    d_loss = distributed_disc_step(data)
                    disc_loss += d_loss
                    num_batch += 1

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print(num_batch)
        disc_loss /= num_batch
        gen_loss /= iterations
        print("Epoch {}, gen_loss  {:.5f} \n disc_loss {:.5f}\n".format(
            epoch, gen_loss, disc_loss))

        generate_and_save_images(generator, epoch, seed)

    generate_and_save_images(generator, FLAGS.epochs, seed)

    anim_file = 'wgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(FLAGS.save_folder +'/image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    embed.embed_file(anim_file)


if __name__ == '__main__':
    app.run(main)
