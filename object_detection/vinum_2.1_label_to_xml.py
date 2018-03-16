from __future__ import print_function
import os
import random
import re
from PIL import Image
import glob
from lxml import etree
import argparse
import json

image_name_idx = 0
image_file_root_to_idx_dict = {}
with open("/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/via_region_data.json") as json_in_file:
    json_file_content = json_in_file.read()
    data_labeled_json_dict = json.loads(json_file_content)
    for image_url in data_labeled_json_dict:
        m = re.match(r".+\/(.+)\.png", image_url)
        original_image_file_root = m.group(1)
        image_file_root_to_idx_dict[original_image_file_root] = image_name_idx
        image_name_idx += 1


def make_one_image_obj_type_xml(obj_type_image_dir,  # to locate the image folder which contains original images
                                obj_type,  # current obj type for example pcr
                                obj_type_image_file_name,
                                region_id_to_region_dict,
                                ):
    annotation_xml = etree.Element('annotation')
    folder_xml = etree.SubElement(annotation_xml, 'folder')
    folder_xml.text = os.path.basename(obj_type_image_dir)
    filename_xml = etree.SubElement(annotation_xml, 'filename')
    filename_xml.text = obj_type_image_file_name
    size_xml = etree.SubElement(annotation_xml, 'size')
    obj_type_image_path = os.path.join(obj_type_image_dir, obj_type_image_file_name)
    assert os.path.exists(obj_type_image_path)
    im = Image.open(obj_type_image_path)
    image_width, image_height = im.size
    width_xml = etree.SubElement(size_xml, 'width')
    width_xml.text = str(image_width)
    height_xml = etree.SubElement(size_xml, 'height')
    height_xml.text = str(image_height)
    depth_xml = etree.SubElement(size_xml, 'depth')
    depth_xml.text = '3'
    segmented_xml = etree.SubElement(annotation_xml, 'segmented')
    segmented_xml.text = '0'
    # for region_id in data_labeled_json_dict[image_url]['regions']:
    for region_id in region_id_to_region_dict:
        # region = data_labeled_json_dict[image_url]['regions'][region_id]
        region = region_id_to_region_dict[region_id]
        region_label = region['region_attributes']['name']
        bbox = region['shape_attributes']
        if region_label == obj_type:
            xmin = str(bbox['x'])
            ymin = str(bbox['y'])
            width = bbox['width']
            height = bbox['height']
            xmax = str(int(xmin) + int(width))
            ymax = str(int(ymin) + int(height))
            print(xmin, ymin, xmax, ymax)
            object_xml = etree.SubElement(annotation_xml, 'object')
            object_name_xml = etree.SubElement(object_xml, 'name')
            object_name_xml.text = region_label
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

    output_xml_content_str = etree.tostring(annotation_xml, pretty_print=True)
    return output_xml_content_str


original_jpg_image_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_03_jpg"
data_for_faster_rcnn_training_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data"
obj_type_image_dir = os.path.join(data_for_faster_rcnn_training_dir, "images")
os.system("mkdir -p " + obj_type_image_dir)
obj_type_annotation_dir = os.path.join(data_for_faster_rcnn_training_dir, "annotations")
# os.system("mkdir -p " + obj_type_annotation_dir)
obj_type_xml_dir = os.path.join(obj_type_annotation_dir, "xmls")
os.system("mkdir -p " + obj_type_xml_dir)
obj_type_image_file_root_array = []
for obj_type in ['pcr']:
    for image_url in data_labeled_json_dict:
        m = re.match(r".+\/(.+)\.png", image_url)
        original_image_file_root = m.group(1)
        obj_type_image_file_root = "%s_%06d" % (obj_type, image_file_root_to_idx_dict[original_image_file_root])
        obj_type_image_file_root_array.append(obj_type_image_file_root)
        obj_type_image_file_path = os.path.join(obj_type_image_dir, obj_type_image_file_root + ".jpg")
        original_image_file_path = os.path.join(original_jpg_image_dir, original_image_file_root + '.jpg')
        if os.path.isfile(obj_type_image_file_path):
            os.unlink(obj_type_image_file_path)
        os.link(original_image_file_path, obj_type_image_file_path)

for obj_type in ['pcr']:
    for image_url in data_labeled_json_dict:
        m = re.match(r".+\/(.+)\.png", image_url)
        original_image_file_root = m.group(1)
        obj_type_image_file_root = "%s_%06d" % (obj_type, image_file_root_to_idx_dict[original_image_file_root])
        obj_type_image_file_name = obj_type_image_file_root + ".jpg"
        xml_file_name = obj_type_image_file_root + '.xml'
        xml_file_path = os.path.join(obj_type_xml_dir, xml_file_name)
        output_xml_content_str = make_one_image_obj_type_xml(obj_type_image_dir,
                                                             obj_type,
                                                             obj_type_image_file_name,
                                                             data_labeled_json_dict[image_url]['regions'])
        with open(xml_file_path, 'w') as f:
            print(output_xml_content_str, file=f)

trainval_list_file_path = os.path.join(obj_type_annotation_dir, "trainval.txt")
with open(trainval_list_file_path, "w") as f:
    for obj_type_image_file_root in sorted(obj_type_image_file_root_array):
        print("%s 1" % obj_type_image_file_root, file=f)

# parser = argparse.ArgumentParser()
#
# parser.add_argument('-O', '--output-dir', action='store', dest='output_dir',
#                     help='output directory to write linked images and xmls of bounding boxes of one obj type.')
#
# parser.add_argument('-I', '--input_dir', action='store', dest='input_dir',
#                     help='input directory MUST contain images, region_<obj_name>_output.txt, optional [obj_name_to_obj_type.txt]')
#
# parser.add_argument('--version', action='version', version='%(prog)s 1.0')
# arg_parse_results = parser.parse_args()
# if arg_parse_results.input_dir is None or arg_parse_results.output_dir is None:
#     parser.print_help()
#     exit(-1)
#
# # output_xml_dir = '/home/gao/Desktop/image_saved_kuka_without_hand/xmls'
# output_separate_dir = arg_parse_results.output_dir
# os.system("mkdir -p " + output_separate_dir)
# assert os.path.isdir(output_separate_dir)
#
# image_dir = arg_parse_results.input_dir
# obj_name_to_obj_type_file_path = image_dir+'/obj_name_to_obj_type.txt'
# if os.path.isfile(obj_name_to_obj_type_file_path):
#     with open(obj_name_to_obj_type_file_path, 'r') as f:
#         obj_name_to_obj_type = dict(map(lambda l: tuple(l.rstrip('\n').lstrip(' ').split(':')), filter(lambda l: len(l.rstrip('\n')) > 0, f.readlines())))
#         # obj_name_to_obj_type = dict(map(lambda l: tuple(l.rstrip('\n').lstrip(' ').split(':')), f.readlines()))
# else:
#     obj_name_to_obj_type = {}
#
# tracking_result_list_path_list = sorted(glob.glob(image_dir+"/region*output.txt"))
# for path in tracking_result_list_path_list:
#     assert os.path.isfile(path)
# obj_type_to_tracking_result_list_path = {}
# for tracking_result_list_path in tracking_result_list_path_list:
#     stem = os.path.basename(tracking_result_list_path).split('.')[0]
#     m = re.match(r"region_(.+)_output", stem)
#     obj_name = m.group(1)
#     if obj_name in obj_name_to_obj_type.keys():
#         obj_type = obj_name_to_obj_type[obj_name]
#     else:
#         obj_type = obj_name
#     if obj_type in obj_type_to_tracking_result_list_path.keys():
#         obj_type_to_tracking_result_list_path[obj_type].append(tracking_result_list_path)
#     else:
#         obj_type_to_tracking_result_list_path[obj_type] = [tracking_result_list_path]

