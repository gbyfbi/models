from __future__ import print_function
import os
import copy
import random
import re
from PIL import Image
from lxml import etree
import glob
data_dir_list = ['/home/gao/Desktop/image_saved_kuka_with_hand', '/home/gao/Desktop/image_saved_kuka_without_hand']
work_dir_for_tf = '/home/gao/Downloads/tensorflow_models_gby/object_detection/kuka_innovation_data'
os.system("mkdir -p " + work_dir_for_tf)
work_image_dir_for_tf = work_dir_for_tf+'/images'
os.system("mkdir -p " + work_image_dir_for_tf)
work_xml_dir_for_tf = work_dir_for_tf+'/annotations/xmls'
os.system("mkdir -p " + work_xml_dir_for_tf)
image_list_file_path_for_tf = work_image_dir_for_tf+"/images.txt"
with open(image_list_file_path_for_tf, "w") as f:
    for data_dir in data_dir_list:
        print("Processing %s." % data_dir)
        if os.path.isfile(data_dir+"/images_downsampled.txt"):
            image_file_list_path = data_dir+"/images_downsampled.txt"
            print("Down sampled image file list is used for %s." % data_dir)
        else:
            image_file_list_path = data_dir+"/images.txt"
        with open(image_file_list_path, "r") as ff:
            image_file_path_list = sorted(map(lambda l: l.rstrip('\n'), ff.readlines()))

        for image_file_path in image_file_path_list:
            image_file_name = os.path.basename(image_file_path)
            image_file_path_for_tf = work_image_dir_for_tf+"/"+image_file_name
            if not os.path.isfile(image_file_path_for_tf):
                os.link(image_file_path, image_file_path_for_tf)
            print(image_file_path_for_tf, file=f)
            image_name = image_file_name.split('.')[0]
            xml_file_path = data_dir+"/xmls/"+image_name+".xml"
            xml_file_path_for_tf = work_xml_dir_for_tf+'/'+image_name+".xml"
            if not os.path.isfile(xml_file_path_for_tf):
                os.link(xml_file_path, xml_file_path_for_tf)

annotation_dir = work_dir_for_tf+"/annotations"
annotation_list_path = os.path.join(annotation_dir, 'list.txt')
annotation_trainval_path = os.path.join(annotation_dir, 'trainval.txt')
annotation_test_path = os.path.join(annotation_dir, 'test.txt')
with open(image_list_file_path_for_tf, 'r') as f:
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
trainval_image_num = int(0.9*image_num)
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

