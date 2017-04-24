import caffe
import numpy as np
import sys
from caffe.proto.caffe_pb2 import BlobProto

# if len(sys.argv) != 3:
#     print ("Usage: python convert_protomean.py proto.mean out.npy")
#     sys.exit()

numpy_mean_file_path = 'imagenet_mean.npy'
data = np.load(numpy_mean_file_path)
print(data.shape)
