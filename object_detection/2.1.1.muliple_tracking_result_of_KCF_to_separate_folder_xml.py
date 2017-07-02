from __future__ import print_function
import os
import random
import re
from PIL import Image
import glob
from lxml import etree
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-O', '--output-dir', action='store', dest='output_dir',
                    help='output directory to write linked images and xmls of bounding boxes of one obj type.')

parser.add_argument('-I', '--input_dir', action='store', dest='input_dir',
                    help='input directory MUST contain images, region_<obj_name>_output.txt, optional [obj_name_to_obj_type.txt]')

parser.add_argument('--version', action='version', version='%(prog)s 1.0')
arg_parse_results = parser.parse_args()
if arg_parse_results.input_dir is None or arg_parse_results.output_dir is None:
    parser.print_help()
    exit(-1)

# output_xml_dir = '/home/gao/Desktop/image_saved_kuka_without_hand/xmls'
output_separate_dir = arg_parse_results.output_dir
os.system("mkdir -p " + output_separate_dir)
assert os.path.isdir(output_separate_dir)

image_dir = arg_parse_results.input_dir
obj_name_to_obj_type_file_path = image_dir+'/obj_name_to_obj_type.txt'
if os.path.isfile(obj_name_to_obj_type_file_path):
    with open(obj_name_to_obj_type_file_path, 'r') as f:
        obj_name_to_obj_type = dict(map(lambda l: tuple(l.rstrip('\n').lstrip(' ').split(':')), filter(lambda l: len(l.rstrip('\n')) > 0, f.readlines())))
        # obj_name_to_obj_type = dict(map(lambda l: tuple(l.rstrip('\n').lstrip(' ').split(':')), f.readlines()))
else:
    obj_name_to_obj_type = {}

tracking_result_list_path_list = sorted(glob.glob(image_dir+"/region*output.txt"))
for path in tracking_result_list_path_list:
    assert os.path.isfile(path)
obj_type_to_tracking_result_list_path = {}
for tracking_result_list_path in tracking_result_list_path_list:
    stem = os.path.basename(tracking_result_list_path).split('.')[0]
    m = re.match(r"region_(.+)_output", stem)
    obj_name = m.group(1)
    if obj_name in obj_name_to_obj_type.keys():
        obj_type = obj_name_to_obj_type[obj_name]
    else:
        obj_type = obj_name
    if obj_type in obj_type_to_tracking_result_list_path.keys():
        obj_type_to_tracking_result_list_path[obj_type].append(tracking_result_list_path)
    else:
        obj_type_to_tracking_result_list_path[obj_type] = [tracking_result_list_path]

# if os.path.isfile(os.path.join(image_dir, "images_downsampled.txt")):
#     image_file_list_path = os.path.join(image_dir, "images_downsampled.txt")
#     print("Down sampled image file list is used for %s." % image_dir)
# else:
#     image_file_list_path = os.path.join(image_dir, "images.txt")
image_file_list_path = os.path.join(image_dir, "images.txt")
assert os.path.isfile(image_file_list_path)
with open(image_file_list_path, 'r') as f:
    image_path_list = map(lambda l: l.rstrip('\n'), f.readlines())
print(image_path_list)

for obj_type in obj_type_to_tracking_result_list_path.keys():
    obj_name_to_tracked_bnd = {}
    obj_type_image_path_list = []
    for tracking_result_list_path in obj_type_to_tracking_result_list_path[obj_type]:
        stem = os.path.basename(tracking_result_list_path).split('.')[0]
        m = re.match(r"region_(.+)_output", stem)
        obj_name = m.group(1)
        with open(tracking_result_list_path, 'r') as f:
            tracking_result_list = map(lambda l: l.rstrip('\n'), f.readlines())
            obj_name_to_tracked_bnd[obj_name] = tracking_result_list
        # print(tracking_result_list)
    obj_type_desc_dir = os.path.join(output_separate_dir, obj_type)
    os.system("mkdir -p "+obj_type_desc_dir)
    obj_type_image_dir = os.path.join(obj_type_desc_dir, "images")
    os.system("mkdir -p "+obj_type_image_dir)
    obj_type_xml_dir = os.path.join(obj_type_desc_dir, "xmls")
    os.system("mkdir -p "+obj_type_xml_dir)

    for idx, image_path in enumerate(image_path_list):
        print(image_path)
        image_file_name = os.path.basename(image_path)
        print(image_file_name)
        m = re.match(r".+_(\d+)\.jpg", image_file_name)
        image_data_idx = m.group(1)
        obj_type_image_stem = obj_type+'_'+image_data_idx
        obj_type_image_file_name = obj_type_image_stem +'.jpg'
        obj_type_image_path = os.path.join(obj_type_image_dir, obj_type_image_file_name)
        if os.path.isfile(obj_type_image_path):
            os.unlink(obj_type_image_path)
        os.link(image_path, obj_type_image_path)
        obj_type_image_path_list.append(obj_type_image_path)

        annotation_xml = etree.Element('annotation')
        folder_xml = etree.SubElement(annotation_xml, 'folder')
        folder_xml.text = os.path.basename(image_dir)
        filename_xml = etree.SubElement(annotation_xml, 'filename')
        filename_xml.text = obj_type_image_file_name
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
        for obj_name in sorted(obj_name_to_tracked_bnd.keys()):
            tracking_result_list = obj_name_to_tracked_bnd[obj_name]
            tracking_result = tracking_result_list[idx]
            xmin, ymin, width, height = tracking_result.split(',')
            print(xmin, ymin, width, height)
            print(type(xmin))
            xmax = str(int(xmin) + int(width))
            ymax = str(int(ymin) + int(height))
            print(xmin, ymin, xmax, ymax)
            object_xml = etree.SubElement(annotation_xml, 'object')
            object_name_xml = etree.SubElement(object_xml, 'name')
            object_name_xml.text = obj_name
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
        xml_file_name = obj_type_image_stem + '.xml'
        xml_file_path = os.path.join(obj_type_xml_dir, xml_file_name)
        with open(xml_file_path, 'w') as f:
            print(output_xml_content_str, file=f)
    obj_type_image_path_list_file_path = os.path.join(obj_type_image_dir, "images.txt")
    with open(obj_type_image_path_list_file_path, "w") as f:
        for obj_type_image_path in sorted(obj_type_image_path_list):
            print(obj_type_image_path, file=f)

