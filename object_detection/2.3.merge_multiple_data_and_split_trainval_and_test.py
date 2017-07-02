from __future__ import print_function
import os
import copy
import random
import re
from PIL import Image
from lxml import etree
import glob
# data_dir_list = ['/home/gao/Desktop/image_saved_kuka_with_hand', '/home/gao/Desktop/image_saved_kuka_without_hand']
separate_obj_type_dir_list = ['/home/gao/Desktop/image_saved_kuka_with_hand/seprate_obj_type_bnd_desc/hand',
                              '/home/gao/Desktop/image_saved_kuka_without_hand/seprate_obj_type_bnd_desc/bolt',
                              '/home/gao/Desktop/image_saved_kuka_without_hand/seprate_obj_type_bnd_desc/meter',
                              '/home/gao/Desktop/image_saved_kuka_without_hand/seprate_obj_type_bnd_desc/mm',
                              '/home/gao/Desktop/image_saved_kuka_without_hand/seprate_obj_type_bnd_desc/slc']
work_dir_for_tf = '/home/gao/Downloads/tensorflow_models_gby/object_detection/kuka_innovation_data'
os.system("mkdir -p " + work_dir_for_tf)
work_image_dir_for_tf = work_dir_for_tf+'/images'
os.system("mkdir -p " + work_image_dir_for_tf)
work_xml_dir_for_tf = work_dir_for_tf+'/annotations/xmls'
os.system("mkdir -p " + work_xml_dir_for_tf)
image_list_file_path_for_tf = work_image_dir_for_tf+"/images.txt"
with open(image_list_file_path_for_tf, "w") as f:
    for obj_type_data_dir in separate_obj_type_dir_list:
        print("Processing %s." % obj_type_data_dir)
        obj_type_image_dir = os.path.join(obj_type_data_dir, 'images')
        obj_type_image_downsampled_list_file_path = os.path.join(obj_type_image_dir, "images_downsampled.txt")
        obj_type_image_list_file_path = os.path.join(obj_type_image_dir, "images.txt")
        if os.path.isfile(obj_type_image_downsampled_list_file_path):
            image_file_list_path = obj_type_image_downsampled_list_file_path
            print("Down sampled image file list is used for %s." % obj_type_data_dir)
        else:
            image_file_list_path = obj_type_image_list_file_path
        with open(image_file_list_path, "r") as ff:
            image_file_path_list = sorted(map(lambda l: l.rstrip('\n'), ff.readlines()))

        for image_file_path in image_file_path_list:
            image_file_name = os.path.basename(image_file_path)
            image_file_path_for_tf = work_image_dir_for_tf+"/"+image_file_name
            if os.path.isfile(image_file_path_for_tf):
                os.unlink(image_file_path_for_tf)
            os.link(image_file_path, image_file_path_for_tf)
            print(image_file_path_for_tf, file=f)
            image_name = image_file_name.split('.')[0]
            xml_file_name = image_name+'.xml'
            xml_file_path = os.path.join(obj_type_data_dir, 'xmls',  xml_file_name)
            xml_file_path_for_tf = os.path.join(work_xml_dir_for_tf, xml_file_name)
            assert os.path.isfile(xml_file_path)
            if os.path.isfile(xml_file_path_for_tf):
                os.unlink(xml_file_path_for_tf)
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
    for image_path in sorted(image_path_shuffled_list[:trainval_image_num]):
        image_file_name = os.path.basename(image_path)
        image_real_name, image_file_ext = os.path.splitext(image_file_name)
        print(image_real_name, 1, file=f)

with open(annotation_test_path, 'w') as f:
    for image_path in sorted(image_path_shuffled_list[trainval_image_num:]):
        image_file_name = os.path.basename(image_path)
        image_real_name, image_file_ext = os.path.splitext(image_file_name)
        print(image_real_name, 1, file=f)

