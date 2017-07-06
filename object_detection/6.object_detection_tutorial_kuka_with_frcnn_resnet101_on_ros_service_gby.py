# coding: utf-8
from __future__ import print_function
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
import threading
import cv2
import rospy
import image_geometry
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo
import math
from timeit import default_timer as timer
import Queue
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg
import time as sys_time
from iiwa_msgs.srv import TensorflowObjectDetection, TensorflowObjectDetectionResponse
from geometry_msgs.msg import Polygon

# What model to download.
TRAINING_DIR = '/home/gao/Downloads/tensorflow_models_gby/object_detection/kuka_innovation_data/training_faster_rcnn_resnet101'
LABEL_DIR = '/home/gao/Downloads/tensorflow_models_gby/object_detection/data'
PATH_TO_CKPT = os.path.join(TRAINING_DIR, 'frozen_inference_graph.pb')
assert os.path.isfile(PATH_TO_CKPT)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(LABEL_DIR, 'kuka_label_map.pbtxt')
NUM_CLASSES = 5

from message_filters import TimeSynchronizer, Subscriber as MsgFilterSubscriber


class ObjectDetector:
    def __init__(self, ):
        self.interThreadQueue = Queue.Queue()
        self.image_index = -1
        self.mutex_image_index = threading.Lock()
        self.processed_image_index = -1
        self.mutex_processed_image_index = threading.Lock()
        self.depth_image_topic_name = "/camera/depth_registered/image_raw"
        self.color_image_topic_name = "/camera/rgb/image_raw"
        self.image_buffer_len = 100
        self.depth_image_buffer = [None] * self.image_buffer_len
        self.color_image_buffer = [None] * self.image_buffer_len
        self.polygon_bnd_buffer = [None] * self.image_buffer_len
        self.cv_bridge = CvBridge()
        self.display_thread = threading.Thread(target=self.__daemon_display_image)
        self.display_thread.start()
        # self.time_synchronizer = TimeSynchronizer((MsgFilterSubscriber(self.color_image_topic_name, RosImage),
        #                                           MsgFilterSubscriber(self.depth_image_topic_name, RosImage)), 10)
        # self.time_synchronizer.registerCallback(self.__callback_on_rgb_and_depth_images)
        # while self.tcp_depth_camera_info_msg is None or self.tcp_depth_pre_computed_3d_rays is None or self.pan_tilt_depth_camera_info_msg is None or self.pan_tilt_depth_pre_computed_3d_rays is None:
        threading.Timer(1, self.nothing, [222]).start()
        threading.Timer(1, self.detect_with_tf, []).start()
        self.obj_detect_srv = rospy.Service('object_detection', TensorflowObjectDetection, self.__obj_detect_srv_cb)

    def __obj_detect_srv_cb(self, req):
        rgb_image_msg = req.rgb_image
        depth_image_msg = req.depth_image
        rgb_cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8")
        depth_cv_image = self.cv_bridge.imgmsg_to_cv2(depth_image_msg, "mono16")
        self.display_image_on_window(rgb_cv_image, "color image")
        self.display_image_on_window(depth_cv_image, "depth image")
        self.mutex_image_index.acquire(True)
        self.image_index += 1
        local_image_index = self.image_index
        self.mutex_image_index.release()
        buffer_index = local_image_index % self.image_buffer_len
        self.color_image_buffer[buffer_index] = rgb_cv_image
        self.depth_image_buffer[buffer_index] = depth_cv_image
        is_finished = False
        while not is_finished:
            sys_time.sleep(0.001)
            self.mutex_image_index.acquire(True)
            self.mutex_processed_image_index.acquire()
            if self.processed_image_index >= local_image_index:
                is_finished = True
            self.mutex_processed_image_index.release()
            self.mutex_image_index.release()
        res = TensorflowObjectDetectionResponse(self.polygon_bnd_buffer[buffer_index])
        return res

    def __callback_on_rgb_and_depth_images(self, rgb_ros_image_msg, depth_ros_image_msg):
        # assert rgb_ros_image_msg.header.stamp == depth_ros_image_msg.header.stamp
        # rgb_cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_ros_image_msg, "bgr8")
        rgb_cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_ros_image_msg, "rgb8")
        depth_cv_image = self.cv_bridge.imgmsg_to_cv2(depth_ros_image_msg, "mono16")
        self.display_image_on_window(rgb_cv_image, "color image")
        self.display_image_on_window(depth_cv_image, "depth image")
        self.mutex_image_index.acquire()
        buffer_index = (self.image_index + 1) % self.image_buffer_len
        self.mutex_image_index.release()
        self.color_image_buffer[buffer_index] = rgb_cv_image
        self.depth_image_buffer[buffer_index] = depth_cv_image
        self.mutex_image_index.acquire(True)
        self.image_index += 1
        self.mutex_image_index.release()

    def display_image_on_window(self, image, window_name, resize_x_axis=1., resize_y_axis=1.):
        self.interThreadQueue.put((window_name, cv2.resize(image, None, fx=resize_x_axis, fy=resize_y_axis)))

    def __daemon_display_image(self):
        while True:
            while self.interThreadQueue.empty() is False:
                window_name, image = self.interThreadQueue.get()
                cv2.imshow(window_name, cv2.resize(image, None, fx=.6, fy=.6))
                cv2.waitKey(10)
            # sys_time.sleep(.01)
            sys_time.sleep(.01)

    def nothing(self, value):
        # print "value:%d" % value
        pass

    def detect_with_tf(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        img_to_show = None
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    self.mutex_image_index.acquire()
                    local_image_index = self.image_index
                    self.mutex_image_index.release()
                    for i_image in range(self.processed_image_index+1, local_image_index+1):
                        buffer_index = i_image % self.image_buffer_len
                        image_np = self.color_image_buffer[buffer_index]
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=2)
                        boxes_np = np.squeeze(boxes)
                        scores_np = np.squeeze(scores)

                        def make_polygon(box):
                            points = [Point32(0, 0, 0), Point32(1, 0, 0), Point32(1, 1, 0), Point32(0, 1, 0)]
                            polygon = Polygon(points)
                            return polygon

                        polygon_msg_list = []
                        for i in range(boxes_np.shape[0]):
                            if scores_np[i] > 0.5:
                                box = tuple(boxes_np[i].tolist())
                                polygon_msg_list.append(make_polygon(box))
                        self.polygon_bnd_buffer[buffer_index] = polygon_msg_list
                        self.mutex_processed_image_index.acquire(True)
                        self.processed_image_index = i_image
                        self.mutex_processed_image_index.release()

                        # plt.figure(figsize=IMAGE_SIZE)
                        if img_to_show is None:
                            img_to_show = plt.imshow(image_np)
                        else:
                            img_to_show.set_data(image_np)
                        plt.pause(.001)
                        plt.draw()
                    # sys_time.sleep(.01)
                    sys_time.sleep(.001)
                    # plt.figure()
                    # plt.imshow(image_np)
                    # plt.show()


rospy.init_node('object_detection_node', anonymous=True)
object_detector = ObjectDetector()
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")


# In[ ]:
