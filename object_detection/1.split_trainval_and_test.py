from __future__ import print_function
import os
import copy
import random
import re
from PIL import Image
from lxml import etree
image_path_list_path = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_data/images/images.txt'
annotation_dir = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_data/annotations'
os.mkdir(annotation_dir)
annotation_list_path = os.path.join(annotation_dir, 'list.txt')
annotation_trainval_path = os.path.join(annotation_dir, 'trainval.txt')
annotation_test_path = os.path.join(annotation_dir, 'test.txt')
with open(image_path_list_path, 'r') as f:
    image_path_list = map(lambda l: l.rstrip('\n'), f.readlines())
print(image_path_list)
image_path_shuffled_list = copy.deepcopy(image_path_list)
random.shuffle(image_path_shuffled_list)
print(image_path_shuffled_list)
with open(annotation_list_path, 'w') as f:
    for image_path in image_path_list:
        image_file_name = os.path.basename(image_path)
        image_real_name, image_file_ext = os.path.splitext(image_file_name)
        print(image_real_name, 1, file=f)

image_num = len(image_path_list)
trainval_image_num = int(0.8*image_num)
test_image_num = image_num - trainval_image_num
with open(annotation_trainval_path, 'w') as f:
    for image_path in image_path_shuffled_list[:trainval_image_num]:
        image_file_name = os.path.basename(image_path)
        image_real_name, image_file_ext = os.path.splitext(image_file_name)
        print(image_real_name, 1, file=f)

with open(annotation_test_path, 'w') as f:
    for image_path in image_path_shuffled_list[trainval_image_num:]:
        image_file_name = os.path.basename(image_path)
        image_real_name, image_file_ext = os.path.splitext(image_file_name)
        print(image_real_name, 1, file=f)

