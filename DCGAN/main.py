"""Main file for training of Deep Convolutional Generative Adversarial Network to generate images"""
import glob
import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from model import Discriminator, Generator

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'lsun/bedroom', 'lsun/bedrooms | mnist | cifar10')
flags.DEFINE_float('lr', 2e-4, 'learning rate')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epochs', 50, 'epochs')
flags.DEFINE_integer('num_examples', 16, 'number of examples to generate')
flags.DEFINE_integer('num_filters', 64, 'number of filters')
flags.DEFINE_integer('image_size', 64, 'size of image')
flags.DEFINE_integer('latent_vector', 100, 'size of latent space')
flags.DEFINE_string('save_folder', 'images', 'folder to save generated images')

def main(_):

    strategy = tf.distribute.MirroredStrategy()        

    if FLAGS.dataset == 'mnist':
        output_channels = 1
    else:
        output_channels = 3

    train_ds, ds_info = tfds.load(
        FLAGS.dataset, split='train', shuffle_files=True, with_info=True)
    
    #dataset is very big, don't want to wait long
    if FLAGS.dataset == 'lsun/bedroom':
        train_ds = train_ds.take(300000)

    def preprocess(image):
        """Normalize the images to [-1.0, 1.0]"""
        image = tf.image.resize_with_pad(image['image'], FLAGS.image_size,
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

        def discriminator_loss(real_output, fake_output):
            real_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
                    tf.ones_like(real_output), real_output)
            
            generated_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
                    tf.zeros_like(fake_output), fake_output)
            
            total_loss = real_loss + generated_loss
            return total_loss

        def generator_loss(fake_output):
            return tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
                    tf.ones_like(fake_output), fake_output)

        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.lr, beta_1=0.5, beta_2=0.999)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.lr, beta_1=0.5, beta_2=0.999)

        inputs = tf.keras.Input(
            shape=(1, 1, FLAGS.latent_vector), name="latent_vector")
        outputs = Generator(FLAGS.num_filters, output_channels)(inputs)
        generator = tf.keras.Model(inputs=inputs, outputs=outputs)

        inputs = tf.keras.Input(
            shape=(FLAGS.image_size, FLAGS.image_size, output_channels),
            name="imgs")
        outputs = Discriminator(FLAGS.num_filters)(inputs)
        discriminator = tf.keras.Model(inputs=inputs, outputs=outputs)


    @tf.function
    def train_step(images):

        noise = tf.random.normal([images.shape[0], 1, 1, FLAGS.latent_vector])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))
        return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss)

    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_gen_loss, per_replica_disc_loss = strategy.run(
            train_step, args=[dist_inputs])
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_gen_loss,
            axis=None), strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_disc_loss, axis=None)

    def save_images(model, ep, vector):

        predictions = tf.clip_by_value(model(vector, training=False), -1, 1)
        fig = plt.figure(figsize=(5, 5))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i].numpy() * 127.5 + 127.5).astype(np.uint8))
            plt.axis('off')

        plt.savefig(FLAGS.save_folder + '/image_at_epoch_{:04d}.png'.format(ep))

    
    if not os.path.exists(FLAGS.save_folder):
         os.makedirs(FLAGS.save_folder)
            
    seed = tf.random.normal([FLAGS.num_examples, 1, 1, FLAGS.latent_vector])
    
    for epoch in tqdm(range(FLAGS.epochs)):
        start = time.time()
        gen_loss = 0
        disc_loss = 0
        num_batch = 0

        for image_batch in train_ds:
            g_loss, d_loss = distributed_train_step(image_batch)
            gen_loss += g_loss
            disc_loss += d_loss
            num_batch += 1

        gen_loss /= num_batch
        disc_loss /= num_batch
        
        print("Epoch {}, gen_loss  {:.5f} \n disc_loss {:.5f}\n".format(
            epoch, gen_loss, disc_loss))

        save_images(generator, epoch, seed)

    save_images(generator, FLAGS.epochs, seed)



if __name__ == '__main__':
    app.run(main)
