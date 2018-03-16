from __future__ import print_function
import os
import re
import json

image_name_idx = 0
# png_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_03"
# png_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_02"
png_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_01"
# jpg_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_03_jpg"
# jpg_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_02_jpg"
jpg_dir = "/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/decoded_01_jpg"
if not os.path.exists(jpg_dir):
    os.system("mkdir -p %s" % jpg_dir)
assert os.path.exists(jpg_dir)
import glob
png_image_file_path_list = sorted(glob.glob(png_dir + "/*.png"))
for png_image_file_path in png_image_file_path_list:
    m = re.match(r".+\/(.+)\.png", png_image_file_path)
    image_file_root = m.group(1)
    jpg_image_file_path = os.path.join(jpg_dir, image_file_root + ".jpg")
    os.system("convert %s %s" % (png_image_file_path, jpg_image_file_path))

# with open("/home/gao/Downloads/tensorflow_models_gby/object_detection/vinum_data/via_region_data.json") as json_in_file:
#     json_file_content = json_in_file.read()
#     data_labeled_json_dict = json.loads(json_file_content)
#     for image_url in data_labeled_json_dict:
#         m = re.match(r".+\/(.+)\.png", image_url)
#         image_file_root = m.group(1)
#         png_image_file_path = os.path.join(png_dir, image_file_root + ".png")
#         jpg_image_file_path = os.path.join(jpg_dir, image_file_root + ".jpg")
#         os.system("convert %s %s" % (png_image_file_path, jpg_image_file_path))

