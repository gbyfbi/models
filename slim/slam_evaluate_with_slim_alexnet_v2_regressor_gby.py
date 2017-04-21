import os
import tensorflow as tf
from datasets import slam
from nets import inception
from nets import alexnet_gby
from preprocessing import no_preprocessing
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.contrib.slim as slim

def load_batch(dataset, batch_size=32, height=224, width=224, is_training=False):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    # image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    image = no_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, images_raw, labels

slam_data_dir = '/home/gao/Data/slam/tf_data/slim/slam_train/slam_tfrecord'
train_dir = '/home/gao/Data/slam/tf_data/slim/slam_train/tf-models/alexnet_v2_gby'
image_size = alexnet_gby.alexnet_v2.default_image_size
batch_size = 1

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    # dataset = flowers.get_split('train', flowers_data_dir)
    # dataset = slam.get_split('validation', slam_data_dir)
    dataset = slam.get_split('train', slam_data_dir)
    images, images_raw, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(alexnet_gby.alexnet_v2_arg_scope()):
        logits, end_points = alexnet_gby.alexnet_v2(images, num_classes=dataset.num_classes, is_training=False)

    # probabilities = tf.nn.softmax(logits)
    probabilities = logits

    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(
      checkpoint_path,
      slim.get_variables_to_restore())

    with tf.variable_scope('alexnet_v2/fc8', reuse=True):
        fc8_biases = tf.get_variable('biases', initializer=tf.constant_initializer(0.0))
        fc8_biases_assign_op = tf.assign(fc8_biases, np.zeros((6,)))

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            # sess.run(fc8_biases_assign_op)
            np_probabilities, np_images_raw, np_labels, np_end_points, np_images, np_variables = sess.run([probabilities, images_raw, labels, end_points, images, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

            for i in xrange(batch_size):
                image = np_images_raw[i, :, :, :]
                true_label = np_labels[i]
                predicted_label = np_probabilities[i, :]
                print(true_label)
                print(predicted_label)
                # predicted_name = dataset.labels_to_names[predicted_label]
                # true_name = dataset.labels_to_names[true_label]

                plt.figure()
                plt.imshow(image.astype(np.uint8))
                # plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                plt.axis('off')
                plt.show()
