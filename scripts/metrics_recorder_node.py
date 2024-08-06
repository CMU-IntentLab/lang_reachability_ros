# !/usr/bin/env python

import os

import rospy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32, String
from nav_msgs.msg import OccupancyGrid

import tf.transformations as tft

import numpy as np
import datetime
import signal
import sys
import argparse
import json

class MetricsRecorderNode:
    def __init__(self, args) -> None:
        self.args = args
        self.exp_config = self.make_exp_config()
        self.topics_names = self.make_topics_names()

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
        self.combined_map_times = []
        self.text_queries = []
        self.vlm_inference_times = []

        self.robot_state_sub = rospy.Subscriber(self.topics_names["pose"], PoseWithCovarianceStamped, callback=self.robot_state_callback)
        self.value_function_sub = rospy.Subscriber(self.topics_names["value_function_at_state"], PoseStamped, callback=self.value_function_callback)
        self.failure_sub = rospy.Subscriber(self.topics_names["failure_set_at_state"], PoseStamped, callback=self.failure_callback)
        self.vlm_inference_time_sub = rospy.Subscriber(self.topics_names["vlm_inference_time"], Float32, callback=self.vlm_inference_time_callback)

        self.constraint_map_sub = rospy.Subscriber(self.topics_names['constraints_grid_map'], OccupancyGrid, callback=self.constraints_map_callback)
        self.nominal_planning_time_sub = rospy.Subscriber(self.topics_names["nominal_planning_time"], Float32, self.nominal_planning_time_callback)
        self.safe_planning_time_sub = rospy.Subscriber(self.topics_names["safe_planning_time"], Float32, callback=self.safe_planning_time_callback)
        self.brt_computation_time_sub = rospy.Subscriber(self.topics_names["brt_computation_time"], Float32, callback=self.brt_computation_time_callback)
        self.grid_map_sub = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, self.grid_map_callback)
        self.semantic_map_sub = rospy.Subscriber(self.topics_names["semantic_grid_map"], OccupancyGrid, self.semantic_map_callback)
        self.language_queries_sub = rospy.Subscriber(self.topics_names["language_constraint"], String, callback=self.text_queries_callback)


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

    def vlm_inference_time_callback(self, msg: Float32):
        self.vlm_inference_times.append([msg.data, self.get_time_since_start()])


    def robot_state_callback(self, msg: PoseWithCovarianceStamped):
        time = self.get_time_since_start()
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        pose = [pos.x, pos.y, euler[2], time]
        self.trajectory.append(pose)

    def value_function_callback(self, msg: PoseStamped):
        time = self.get_time_since_start()
        value = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, time]
        self.value_function_at_state.append(value)
    
    def failure_callback(self, msg: PoseStamped):
        time = self.get_time_since_start()
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
        self.combined_map_times.append((self.get_time_since_start(), msg.info.resolution,
                                        (msg.info.origin.position.x, msg.info.origin.position.y)))

    def semantic_map_callback(self, msg: OccupancyGrid):
        semantic_map_now = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.semantic_map_over_time is None:
            self.semantic_map_over_time = semantic_map_now
        else:
            self.semantic_map_over_time = np.dstack((self.semantic_map_over_time, semantic_map_now))
        self.semantic_map_times.append((self.get_time_since_start(), msg.info.resolution,
                                        (msg.info.origin.position.x, msg.info.origin.position.y)))

    def nominal_planning_time_callback(self, msg: Float32):
        self.nominal_planning_time.append([msg.data, self.get_time_since_start()])

    def safe_planning_time_callback(self, msg: Float32):
        self.safe_planning_time.append([msg.data, self.get_time_since_start()])

    def brt_computation_time_callback(self, msg: Float32):
        self.brt_computation_time.append([msg.data, self.get_time_since_start()])

    def text_queries_callback(self, msg: String):
        query = msg.data
        self.text_queries.append(query)

    def save_all_metrics(self):
        print("saving all metrics")
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
        dtype = [('time', 'i4'), ('resolution', 'f8'), ('origin', '2f8')]
        np.save(os.path.join(self.save_path, now, "semantic_map_times.npy"), np.array(self.semantic_map_times, dtype=dtype))
        np.save(os.path.join(self.save_path, now, "combined_map_times.npy"), np.array(self.combined_map_times, dtype=dtype))
        np.save(os.path.join(self.save_path, now, "vlm_inference_times.npy"), self.vlm_inference_times)

        with open(os.path.join(self.save_path, now, "text_queries.txt"), "w") as file:
            for query in self.text_queries:
                file.write(query + "\n")
        with open(os.path.join(self.save_path, now, "exp_config.json"), "w") as file:
            json.dump(self.exp_config, file)

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
print("Started recording data")

def signal_handler(signal, frame):
    print("Terminating and saving metrics...")
    node.save_all_metrics()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

last_print = rospy.Time.now().secs
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    rate.sleep()

    if rospy.Time.now().secs - last_print > 5:
       last_print = rospy.Time.now().secs 
       print("I'm still alive and recording data!")
