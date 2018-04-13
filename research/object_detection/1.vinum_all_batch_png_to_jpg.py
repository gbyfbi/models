from __future__ import print_function
import os
import re
import json

# image_name_idx = 0
batch_data_folder = "/exchange/Dataset/UniCatt/2018-02-14/data/labeled_data"
batch_prefix = "batch"
batch_index_list = [0, 1, 2, 3, 5, 6]
# image_file_root_and_batch_idx_to_idx_dict = {}
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
            png_image_file_path = os.path.join(cur_batch_folder, image_file_root + ".png")
            jpg_image_file_path = os.path.join(cur_batch_folder, image_file_root + ".jpg")
            if not os.path.exists(jpg_image_file_path):
                os.system("convert %s %s" % (png_image_file_path, jpg_image_file_path))
            assert(os.path.exists(jpg_image_file_path))
            # image_file_root_and_batch_idx_to_idx_dict[batch_index][image_file_root] = image_name_idx
            # image_name_idx += 1
