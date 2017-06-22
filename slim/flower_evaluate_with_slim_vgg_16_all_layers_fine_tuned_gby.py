import os
import tensorflow as tf
from datasets import flowers, dataset_utils
from nets import vgg
from preprocessing import vgg_preprocessing
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.contrib.slim as slim

def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
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
    image = vgg_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

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

flowers_data_dir = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/flowers'
train_dir = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/flowers-models/vgg16_debug/all_layers'
# train_dir = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/flowers-models/alexnet_original_finetuned_all_layers'
image_size = vgg.vgg_16.default_image_size
batch_size = 32

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    # dataset = flowers.get_split('train', flowers_data_dir)
    dataset = flowers.get_split('validation', flowers_data_dir)
    images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(images, num_classes=dataset.num_classes, is_training=False)

    probabilities = tf.nn.softmax(logits)

    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(
      checkpoint_path,
      slim.get_variables_to_restore())

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            # sess.run(tf.initialize_local_variables())
            init_fn(sess)
            np_probabilities, np_images_raw, np_labels, np_end_points, np_images = sess.run([probabilities, images_raw, labels, end_points, images])

            for i in xrange(batch_size):
                image = np_images_raw[i, :, :, :]
                true_label = np_labels[i]
                predicted_label = np.argmax(np_probabilities[i, :])
                predicted_name = dataset.labels_to_names[predicted_label]
                true_name = dataset.labels_to_names[true_label]

                plt.figure()
                plt.imshow(image.astype(np.uint8))
                plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                plt.axis('off')
                plt.show()