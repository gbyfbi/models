import caffe
import numpy as np
import sys
from caffe.proto.caffe_pb2 import BlobProto

# if len(sys.argv) != 3:
#     print ("Usage: python convert_protomean.py proto.mean out.npy")
#     sys.exit()

proto_mean_file_path = 'preprocessing/imagenet_mean.binaryproto'
numpy_mean_file_path = 'preprocessing/imagenet_mean.npy'
blob = BlobProto()
data = open(proto_mean_file_path, 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save(numpy_mean_file_path, out)
