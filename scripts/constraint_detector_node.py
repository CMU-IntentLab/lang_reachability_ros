#!/usr/bin/env python3

import sys
sys.path.append("/home/riss_ws/src/lang_reachability_ros/scripts/lang-reachability")

import rospy
from sensor_msgs.msg import CameraInfo, Image

import numpy as np

import lang_reachability

# TODO:
# - for now I will just publish the camera and depth output; will figure out 
#   how to deal with owl-vit detections later

class ConstraintDetectorNode:
    def __init__(self) -> None:
        self.camera_info_pub = rospy.Publisher("rgb/camera_info", CameraInfo, queue_size=10)
        self.rgb_pub = rospy.Publisher("rgb/image", Image, queue_size=10)
        self.depth_pub = rospy.Publisher("depth/image", Image, queue_size=10)

    def publish_camera_info(self, camera_info: np.array):
        pass

    def publish_rgb(self, rgb_img: np.array):
        pass

    def publish_depth(self, depth_img: np.array):
        pass

    def publish_detection(self):
        pass

rospy.init_node("constraint_detector_node")
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    node = ConstraintDetectorNode()
    rate.sleep()
