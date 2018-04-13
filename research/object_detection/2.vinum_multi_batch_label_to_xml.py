from __future__ import print_function
import os
import random
import re
from PIL import Image
import glob
from lxml import etree
import argparse
import json


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
    num_region_recorded = 0
    for region_id in region_id_to_region_dict:
        # region = data_labeled_json_dict[image_url]['regions'][region_id]
        region = region_id_to_region_dict[region_id]
        if 'name' not in region['region_attributes']:
            continue
        region_label = region['region_attributes']['name']
        bbox = region['shape_attributes']
        if region_label == obj_type:
            num_region_recorded += 1
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

    if num_region_recorded < 1:
        return ''
    output_xml_content_str = etree.tostring(annotation_xml, pretty_print=True)
    return output_xml_content_str


image_name_idx = 0
batch_data_folder = "/exchange/Dataset/UniCatt/2018-02-14/data/labeled_data"
batch_prefix = "batch"
batch_index_list = [0, 1, 2, 3, 5, 6]

data_for_faster_rcnn_training_dir = "/exchange/DL_working_dir/object_detection/tf_faster_rcnn/vinum_data"
obj_type_image_dir = os.path.join(data_for_faster_rcnn_training_dir, "images")
os.system("mkdir -p " + obj_type_image_dir)
obj_type_annotation_dir = os.path.join(data_for_faster_rcnn_training_dir, "annotations")
# os.system("mkdir -p " + obj_type_annotation_dir)
obj_type_xml_dir = os.path.join(obj_type_annotation_dir, "xmls")
os.system("mkdir -p " + obj_type_xml_dir)
obj_type_image_file_root_array = []

for obj_type in ['ppr']:
    for batch_index in batch_index_list:
        batch_name = "%s_%04d" % (batch_prefix, batch_index)
        cur_batch_folder = os.path.join(batch_data_folder, batch_name)
        assert os.path.isdir(cur_batch_folder)
        cur_batch_label_json_path = os.path.join(cur_batch_folder, "label_data.json")
        assert os.path.exists(cur_batch_label_json_path)
        with open(cur_batch_label_json_path) as json_in_file:
            json_file_content = json_in_file.read()
            data_labeled_dict = json.loads(json_file_content)
            for image_url in data_labeled_dict:
                m = re.match(r".+\/(.+)\.png", image_url)
                image_file_root = m.group(1)
                jpg_image_file_path = os.path.join(cur_batch_folder, image_file_root + ".jpg")
                assert (os.path.exists(jpg_image_file_path))
                obj_type_image_file_root = "%s_%06d" % (obj_type, image_name_idx)
                obj_type_image_file_path = os.path.join(obj_type_image_dir, obj_type_image_file_root + ".jpg")
                obj_type_image_file_name = obj_type_image_file_root + ".jpg"
                if os.path.isfile(obj_type_image_file_path):
                    os.unlink(obj_type_image_file_path)
                os.link(jpg_image_file_path, obj_type_image_file_path)
                xml_file_name = obj_type_image_file_root + '.xml'
                xml_file_path = os.path.join(obj_type_xml_dir, xml_file_name)
                output_xml_content_str = make_one_image_obj_type_xml(obj_type_image_dir,
                                                                     obj_type,
                                                                     obj_type_image_file_name,
                                                                     data_labeled_dict[image_url]['regions'])

                with open(xml_file_path, 'w') as f:
                    print(output_xml_content_str, file=f)

                if output_xml_content_str != '':
                    obj_type_image_file_root_array.append(obj_type_image_file_root)

                image_name_idx += 1

trainval_list_file_path = os.path.join(obj_type_annotation_dir, "trainval.txt")
with open(trainval_list_file_path, "w") as f:
    for obj_type_image_file_root in sorted(obj_type_image_file_root_array):
        print("%s 1" % obj_type_image_file_root, file=f)
