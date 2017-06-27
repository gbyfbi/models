
# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
IMAGE_DIR = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_data/images'
IMAGE_EXT = '.jpg'
TRAINING_DIR = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_data/training'
LABEL_DIR = '/home/gao/Downloads/tensorflow_models_gby/object_detection/data'
PATH_TO_CKPT = os.path.join(TRAINING_DIR, 'frozen_inference_graph.pb')
assert os.path.isfile(PATH_TO_CKPT)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(LABEL_DIR, 'switch_label_map.pbtxt')

# NUM_CLASSES = 90
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
import glob
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+"/*.*g")
with open('/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_data/annotations/test.txt', 'r') as f:
    TEST_IMAGE_PATHS = map(lambda l: os.path.join(IMAGE_DIR, l.split(' ')[0]+IMAGE_EXT), f.readlines())
import random
random.shuffle(TEST_IMAGE_PATHS)
TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[:30]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:


img_to_show = None
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      print(image_path)
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      # plt.figure(figsize=IMAGE_SIZE)
      if img_to_show is None:
          img_to_show = plt.imshow(image_np)
      else:
          img_to_show.set_data(image_np)
      plt.pause(.001)
      plt.draw()
      # plt.figure()
      # plt.imshow(image_np)
      # plt.show()


# In[ ]:




