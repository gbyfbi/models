# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils
import csv

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://download.tensorflow.org/example_images/slam_photos.tgz'

# The number of images in the validation set.
_NUM_VALIDATION = 1

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 1


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_file_paths_and_filename2id_dict(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  slam_photo_dir= os.path.join(dataset_dir, 'slam_photos')
  photo_paths = tf.gfile.Glob(slam_photo_dir+'/*.jpg')
  # for filename in os.listdir(slam_photo_dir):
  #   path = os.path.join(slam_photo_dir, filename)
  #   photo_paths.append(path)

  filename_and_xyzabc_csv_path = os.path.join(dataset_dir, 'filename_and_xyzabc.csv')
  with open(filename_and_xyzabc_csv_path, 'rb') as f:
    reader = csv.reader(f)
    filename_and_xyzabc_list = list(reader)

  filename_to_xyzabc_dict = {}
  for filename_and_xyzabc in filename_and_xyzabc_list:
    filename_to_xyzabc_dict[filename_and_xyzabc[0]] = map(lambda x: float(x), filename_and_xyzabc[1:])
  return photo_paths, filename_to_xyzabc_dict


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'slam_%s_%05d-of-%05d.tfrecord' % (
    split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, file_paths, file_name_to_ids, dataset_dir, output_tfrecord_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    file_paths: A list of absolute paths to png or jpg images.
    file_name_to_ids: A dictionary from file names (strings) to ids
      [x, y, z, a, b, c].
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(file_paths) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
          output_tfrecord_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(file_paths))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
              i + 1, len(file_paths), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(file_paths[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            # class_name = os.path.basename(os.path.dirname(filenames[i]))
            file_name = os.path.basename(file_paths[i])
            class_id = file_name_to_ids[file_name]

            example = dataset_utils.image_to_tfexample_regression(
              image_data, 'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'slam_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
        dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, output_tfrecord_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  # if not tf.gfile.Exists(dataset_dir):
  #   tf.gfile.MakeDirs(dataset_dir)

  # if _dataset_exists(dataset_dir):
  #   print('Dataset files already exist. Exiting without re-creating them.')
  #   return

  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_paths, file_names_to_ids = _get_file_paths_and_filename2id_dict(dataset_dir)

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_paths)
  training_filenames = photo_paths[_NUM_VALIDATION:]
  validation_filenames = photo_paths[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, file_names_to_ids,
                   dataset_dir, output_tfrecord_dir)
  _convert_dataset('validation', validation_filenames, file_names_to_ids,
                   dataset_dir, output_tfrecord_dir)

  # Finally, write the labels file:
  # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')


if __name__ == '__main__':
  run('/home/gao/Data/slam/original_data', '/home/gao/Data/slam/tf_data/slim/slam_train/slam_tfrecord')
