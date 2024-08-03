# !/usr/bin/env python

import os

import rospy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32, String
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image

import cv2
import cv_bridge
import tf.transformations as tft

import numpy as np
import datetime

import argparse
import json

class MetricsRecorderNode:
    def __init__(self, args) -> None:
        self.args = args
        self.exp_config = self.make_exp_config()
        self.topics_names = self.make_topics_names()
        self.bridge = cv_bridge.CvBridge()

        self.save_path = self.exp_config["results_path"]
        self.start_time = rospy.Time.now().secs

        self.trajectory = []
        self.value_function_at_state = []
        self.failure_at_state = []
        self.nominal_planning_time = []
        self.safe_planning_time = []
        self.brt_computation_time = []
        self.map_size_meters = None
        self.floorplan = None
        self.combined_map_over_time = None
        self.semantic_map_over_time = None
        self.semantic_map_times = []
        self.combined_times = []
        self.text_queries = []
        self.rgb_images = None
        self.rgb_detections = None

        self.robot_state_sub = rospy.Subscriber(self.topics_names["pose"], PoseWithCovarianceStamped, callback=self.robot_state_callback)
        self.value_function_sub = rospy.Subscriber(self.topics_names["value_function_at_state"], PoseStamped, callback=self.value_function_callback)
        self.failure_sub = rospy.Subscriber(self.topics_names["failure_set_at_state"], PoseStamped, callback=self.failure_callback)
        # self.planning_latency_sub = rospy.Subscriber()
        self.nominal_planning_time_sub = rospy.Subscriber(self.topics_names["nominal_planning_time"], Float32, self.nominal_planning_time_callback)
        self.safe_planning_time_sub = rospy.Subscriber(self.topics_names["safe_planning_time"], Float32, callback=self.safe_planning_time_callback)
        self.brt_computation_time_sub = rospy.Subscriber(self.topics_names["brt_computation_time"], Float32, callback=self.brt_computation_time_callback)
        self.grid_map_sub = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, self.grid_map_callback)
        self.semantic_map_sub = rospy.Subscriber(self.topics_names["semantic_grid_map"], OccupancyGrid, self.semantic_map_callback)
        self.language_queries_sub = rospy.Subscriber(self.topics_names["language_constraint"], String, callback=self.text_queries_callback)
        self.rgb_image_sub = rospy.Subscriber(self.topics_names["rgb_image"], Image, callback=self.rgb_image_callback)
        self.rgb_detections_sub = rospy.Subscriber(self.topics_names["vlm_detections"], Image, callback=self.rgb_detection_callback)

    def make_exp_config(self):
        self.exp_path = self.args.exp_path
        with open(self.exp_path, 'r') as f:
            exp_config = json.load(f)
        return exp_config

    def make_topics_names(self):
        self.topics_path = self.args.topics_path
        with open(self.topics_path, 'r') as f:
            topics_names = json.load(f)
        return topics_names
    
    def robot_state_callback(self, msg: PoseWithCovarianceStamped):
        time = msg.header.stamp.secs
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        pose = [pos.x, pos.y, euler[2], time]
        self.trajectory.append(pose)

    def value_function_callback(self, msg: PoseStamped):
        time = msg.header.stamp.secs
        value = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, time]
        self.value_function_at_state.append(value)
    
    def failure_callback(self, msg: PoseStamped):
        time = msg.header.stamp.secs
        value = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, time]
        self.failure_at_state.append(value)

    def grid_map_callback(self, msg: OccupancyGrid):
        if self.floorplan is None:
            self.floorplan = np.reshape(msg.data, (msg.info.height, msg.info.width))

        if self.map_size_meters is None:
            bottom_left = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
            top_right = bottom_left + np.array([msg.info.width, msg.info.height]) * msg.info.resolution
            self.map_size_meters = np.vstack((bottom_left, top_right))

    def constraints_map_callback(self, msg: OccupancyGrid):
        combined_map_now = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.combined_map_over_time is None:
            self.combined_map_over_time = combined_map_now
        else:
            self.combined_map_over_time = np.dstack((self.combined_map_over_time, combined_map_now))
        self.combined_map_times.append(self.get_time_since_start())

    def semantic_map_callback(self, msg: OccupancyGrid):
        semantic_map_now = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.semantic_map_over_time is None:
            self.semantic_map_over_time = semantic_map_now
        else:
            self.semantic_map_over_time = np.dstack((self.semantic_map_over_time, semantic_map_now))
        self.semantic_map_times.append(self.get_time_since_start())

    def nominal_planning_time_callback(self, msg: Float32):
        self.nominal_planning_time.append([msg.data, self.get_time_since_start()])

    def safe_planning_time_callback(self, msg: Float32):
        self.safe_planning_time.append([msg.data, self.get_time_since_start()])

    def brt_computation_time_callback(self, msg: Float32):
        self.brt_computation_time.append([msg.data, self.get_time_since_start()])

    def text_queries_callback(self, msg: String):
        query = msg.data
        self.text_queries.append(query)

    def rgb_image_callback(self, msg: Image):
        img_array = np.array(self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))
        if self.rgb_images is None:
            self.rgb_images = img_array
        else:
            self.rgb_images = np.dstack((self.rgb_images, img_array))

    def rgb_detection_callback(self, msg: Image):
        img_array = np.array(self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))
        if self.rgb_images is None:
            self.rgb_detections = img_array
        else:
            self.rgb_detections = np.dstack((self.rgb_images, img_array))

    def save_all_metrics(self):
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H:%M:%S")
        os.makedirs(os.path.join(self.save_path, now))
        np.save(os.path.join(self.save_path, now, "trajectory.npy"), self.trajectory)
        np.save(os.path.join(self.save_path, now, "value_function_at_state.npy"), self.value_function_at_state)
        np.save(os.path.join(self.save_path, now, "failure_at_state.npy"), self.failure_at_state)
        np.save(os.path.join(self.save_path, now, "nominal_planning_time.npy"), self.nominal_planning_time)
        np.save(os.path.join(self.save_path, now, "safe_planning_time.npy"), self.safe_planning_time)
        np.save(os.path.join(self.save_path, now, "brt_computation_time.npy"), self.brt_computation_time)
        np.save(os.path.join(self.save_path, now, "map_size_meters.npy"), self.map_size_meters)
        np.save(os.path.join(self.save_path, now, "floorplan.npy"), self.floorplan)
        np.save(os.path.join(self.save_path, now, "combined_map_over_time.npy"), self.combined_map_over_time)
        np.save(os.path.join(self.save_path, now, "semantic_map_over_time.npy"), self.semantic_map_over_time)
        np.save(os.path.join(self.save_path, now, "semantic_map_times.npy"), self.semantic_map_times)
        np.save(os.path.join(self.save_path, now, "combined_times.npy"), self.combined_times)

        with open(os.path.join(self.save_path, now, "text_queries.txt"), "w") as file:
            for query in self.text_queries:
                file.write(query + "\n")

        size = np.shape(self.rgb_images)
        print(f"video size = {size}")
        out = cv2.VideoWriter(os.path.join(self.save_path, now, 'rgb_images.avi'), 0, 30, (size[1], size[0]), False)
        for frame in self.rgb_images:
            out.write(frame)
        # out.release()

        size = np.shape(self.rgb_detections)
        out = cv2.VideoWriter(os.path.join(self.save_path, now, 'rgb_detections.avi'), 0, 30, (size[1], size[0]), False)
        for frame in self.rgb_detections:
            out.write(frame)
        # out.release()

    def get_time_since_start(self):
        return rospy.Time.now().secs - self.start_time


parser = argparse.ArgumentParser(description="Command Node")
parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
args = parser.parse_args()

assert args.exp_path is not None, "a experiment config file must be provided"
assert args.topics_path is not None, "topics names json file must be provided"

rospy.init_node("metrics_recorder_node")
node = MetricsRecorderNode(args)
print("Started recording data!")

last_print = rospy.Time.now().secs
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    rate.sleep()

    if rospy.Time.now().secs - last_print > 5:
       last_print = rospy.Time.now().secs 
       print("I'm still alive and recording data!")
       
node.save_all_metrics()