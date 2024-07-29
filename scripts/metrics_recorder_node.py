# !/usr/bin/env python

import os

import rospy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32, String
from nav_msgs.msg import OccupancyGrid

import tf.transformations as tft

import numpy as np

class MetricsRecorderNode:
    def __init__(self, save_path) -> None:
        self.save_path = save_path

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
        self.text_queries = []

        self.robot_state_sub = rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped, callback=self.robot_state_callback)
        self.value_function_sub = rospy.Subscriber("value_function_at_state", PoseStamped, callback=self.value_function_callback)
        self.failure_sub = rospy.Subscriber("failure_at_state", PoseStamped, callback=self.failure_callback)
        # self.safety_override_sub = rospy.Subscriber()
        self.planning_latency_sub = rospy.Subscriber()
        self.nominal_planning_time_sub = rospy.Subscriber("nominal_planning_time", Float32, self.nominal_planning_time_callback)
        self.safe_planning_time_sub = rospy.Subscriber("safe_planning_time", Float32, callback=self.safe_planning_time_callback)
        self.brt_computation_time_sub = rospy.Subscriber("brt_computation_time", Float32, callback=self.brt_computation_time_callback)
        self.grid_map_sub = rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid, self.grid_map_callback)
        self.semantic_map_sub = rospy.Subscriber("semantic_grid_map", OccupancyGrid, self.semantic_map_callback)
        self.language_queries_sub = rospy.Subscriber("text_queries", String, callback=self.text_queries_callback)

    def robot_state_callback(self, msg: PoseWithCovarianceStamped):
        time = msg.header.stamp.secs
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat[0], quat[1], quat[2], quat[3]])
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

        if self.combined_map_over_time is None:
            self.combined_map_over_time = self.floorplan
        else:
            combined_map_now = np.reshape(msg.data, (msg.info.height, msg.info.width))
            self.combined_map_over_time = np.stack((self.combined_map_over_time, combined_map_now), axis=-1)

    def nominal_planning_time_callback(self, msg: Float32):
        self.nominal_planning_time.append([msg.data, rospy.Time.now().secs])

    def safe_planning_time_callback(self, msg: Float32):
        self.safe_planning_time.append([msg.data, rospy.Time.now().secs])

    def brt_computation_time_callback(self, msg: Float32):
        self.brt_computation_time.append([msg.data, rospy.Time.now().secs])

    def semantic_map_callback(self, msg: OccupancyGrid):
        semantic_map_now = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.semantic_map_over_time is None:
            self.semantic_map_over_time = semantic_map_now
        else:
            self.semantic_map_over_time = np.stack((self.semantic_map_over_time, semantic_map_now), axis=-1)
        self.semantic_map_times.append(rospy.Time.now().secs)

    def text_queries_callback(self, msg: String):
        query = msg.data
        self.text_queries.append(query)

    def save_all_metrics(self):
        np.save(os.path.join(self.save_path, "trajectory.npy"), self.trajectory)
        np.save(os.path.join(self.save_path, "value_function_at_state.npy"), self.value_function_at_state)
        np.save(os.path.join(self.save_path, "failure_at_state.npy"), self.failure_at_state)
        np.save(os.path.join(self.save_path, "nominal_planning_time.npy"), self.nominal_planning_time)
        np.save(os.path.join(self.save_path, "safe_planning_time.npy"), self.safe_planning_time)
        np.save(os.path.join(self.save_path, "brt_computation_time.npy"), self.brt_computation_time)
        np.save(os.path.join(self.save_path, "map_size_meters.npy"), self.map_size_meters)
        np.save(os.path.join(self.save_path, "floorplan.npy"), self.floorplan)
        np.save(os.path.join(self.save_path, "combined_map_over_time.npy"), self.combined_map_over_time)
        np.save(os.path.join(self.save_path, "semantic_map_over_time.npy"), self.semantic_map_over_time)
        np.save(os.path.join(self.save_path, "semantic_map_times.npy"), self.semantic_map_times)

        with open(os.path.join(self.save_path, "text_queries.txt"), "w") as file:
            for query in self.text_queries:
                file.write(query + "\n")


rospy.init_node("metrics_recorder_node")
node = MetricsRecorderNode()

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    rate.sleep()

node.save_all_metrics()