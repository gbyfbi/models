from __future__ import print_function
import os
import random
import re
from PIL import Image
import glob
from lxml import etree
import argparse
import numpy as np

parser = argparse.ArgumentParser()

# parser.add_argument('-O', '--output-list', action='store', dest='output_dir',
#                     help='output directory to write xml files.')

parser.add_argument('-K', '--downsample-factor', action='store', dest='keep_factor', type=float,
                    help='the potion to keep when down-sampling.')

parser.add_argument('-I', '--input_dir', action='store', dest='input_dir',
                    help='input directory containing images, region_xxx_output.txt, obj_name->obj_type etc.')

parser.add_argument('--version', action='version', version='%(prog)s 1.0')
arg_parse_results = parser.parse_args()
if arg_parse_results.input_dir is None or arg_parse_results.keep_factor is None:
    parser.print_help()
    exit(-1)

image_dir = arg_parse_results.input_dir
image_path_list_path = image_dir+'/images.txt'
assert os.path.isfile(image_path_list_path)
with open(image_path_list_path, 'r') as f:
    image_path_list = map(lambda l: l.rstrip('\n'), f.readlines())
print(image_path_list)
image_num = len(image_path_list)
image_num_to_keep = int(image_num * arg_parse_results.keep_factor)
image_idx_to_keep = np.unique(np.rint(np.linspace(0, image_num, image_num_to_keep, endpoint=False)).astype(np.int32))
downsample_image_list_path = image_dir+"/images_downsampled.txt"
downsample_image_list = map(lambda i: image_path_list[i], image_idx_to_keep)
with open(downsample_image_list_path, "w") as f:
    for image_path in downsample_image_list:
        print(image_path, file=f)


