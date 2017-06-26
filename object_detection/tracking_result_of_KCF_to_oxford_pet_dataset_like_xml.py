from __future__ import print_function
import os
import random
import re
from PIL import Image

from lxml import etree
image_path_list_path = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_images/images.txt'
tracking_result_list_path = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_images/output.txt'
output_xml_dir = '/home/gao/Downloads/tensorflow_models_gby/object_detection/collimator_switch_xmls'
os.mkdir(output_xml_dir)
with open(image_path_list_path, 'r') as f:
    image_path_list = map(lambda l: l.rstrip('\n'), f.readlines())
print(image_path_list)
with open(tracking_result_list_path, 'r') as f:
    tracking_result_list = map(lambda l: l.rstrip('\n'), f.readlines())
print(tracking_result_list)
assert len(tracking_result_list) == len(image_path_list)
for image_path, tracking_result in zip(image_path_list, tracking_result_list):
    print(image_path, tracking_result)
    image_file_name = os.path.basename(image_path)
    print(image_file_name)
    xmin, ymin, width, height = tracking_result.split(',')
    print(xmin, ymin, width, height)
    print(type(xmin))
    xmax = str(int(xmin) + int(width))
    ymax = str(int(ymin) + int(height))
    print(xmin, ymin, xmax, ymax)
    annotation_xml = etree.Element('Annotation')
    folder_xml = etree.SubElement(annotation_xml, 'folder')
    folder_xml.text = 'collimator_switch'
    filename_xml = etree.SubElement(annotation_xml, 'filename')
    filename_xml.text = image_file_name
    size_xml = etree.SubElement(annotation_xml, 'size')
    im = Image.open(image_path)
    image_width, image_height = im.size
    width_xml = etree.SubElement(size_xml, 'width')
    width_xml.text = str(image_width)
    height_xml = etree.SubElement(size_xml, 'height')
    height_xml.text = str(image_height)
    depth_xml = etree.SubElement(size_xml, 'depth')
    depth_xml.text = '3'
    segmented_xml = etree.SubElement(annotation_xml, 'segmented')
    segmented_xml.text = '0'
    object_xml = etree.SubElement(annotation_xml, 'object')
    object_name_xml = etree.SubElement(object_xml, 'name')
    object_name_xml.text = 'switch'
    object_pose_xml = etree.SubElement(object_xml, 'pose')
    object_pose_xml.text = 'Frontal'
    object_truncated_xml = etree.SubElement(object_xml, 'truncated')
    object_truncated_xml.text = '0'
    object_occluded_xml = etree.SubElement(object_xml, 'occluded')
    object_occluded_xml.text = '0'
    bndbox_xml = etree.SubElement(object_xml, 'bndbox')
    xmin_xml = etree.SubElement(bndbox_xml, 'xmin')
    xmin_xml.text = xmin
    ymin_xml = etree.SubElement(bndbox_xml, 'ymin')
    ymin_xml.text = ymin
    xmax_xml = etree.SubElement(bndbox_xml, 'xmax')
    xmax_xml.text = xmax
    ymax_xml = etree.SubElement(bndbox_xml, 'ymax')
    ymax_xml.text = ymax
    difficult_xml = etree.SubElement(object_xml, 'difficult')
    difficult_xml.text = '0'
    output_xml_str = etree.tostring(annotation_xml, pretty_print=True)

    print(image_width, image_height)
    image_real_name, image_file_ext = os.path.splitext(image_file_name)
    print(image_real_name, image_file_ext)
    xml_file_path = output_xml_dir + '/' + image_real_name + '.xml'
    with open(xml_file_path, 'w') as f:
        print(output_xml_str, file=f)



