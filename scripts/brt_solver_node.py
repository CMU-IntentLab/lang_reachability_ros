# !/usr/bin/env python
import argparse
import json
import os
import rospy
import threading
import time

from geometry_msgs.msg import Twist, TwistStamped, PoseWithCovarianceStamped, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension

import tf2_ros
import tf.transformations as tft 

import numpy as np

from lang_reachability import reachability


class BRTSolverNode:
    def __init__(self, args) -> None:
        self.args = args
        self.exp_config = self.make_exp_config()
        self.topics_names = self.make_topics_names()

        self.reachability_solver = None
        self.robot_pose = None
        self.map_resolution = -1
        self.map_origin = None
        self.grid_map = None
        self.semantic_grid_map = None
        self.last_updated = -1
        self.brt_computed = False
        self.values = None

        self.use_precomputed_brt = self.exp_config['use_precomputed_brt'] == "True"
        self.scene_name = self.exp_config['scene_name']
        self.vmin = self.exp_config["vmin"]
        self.vmax = self.exp_config["vmax"]
        self.wmax = self.exp_config["wmax"]
        self.brt_update_interval = self.exp_config["brt_update_interval"]
        self.epsilon = self.exp_config["brt_convergence_epsilon"]
        self.unsafe_level = self.exp_config["brt_unsafe_level"]

        self.possible_brt_file = os.path.join(self.exp_config["results_path"], f"brt_{self.scene_name}.npy")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot_pose_sub = rospy.Subscriber(self.topics_names["pose"], PoseWithCovarianceStamped, callback=self.robot_pose_callback)
        # self.robot_odom_sub = rospy.Subscriber(self.topics_names["odom"], Odometry, callback=self.odom_callback)
        self.floorplan_sub = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, callback=self.floorplan_callback)
        self.semantic_map_sub = rospy.Subscriber(self.topics_names["semantic_grid_map"], OccupancyGrid, callback=self.semantic_map_callback)

        self.brt_pub = rospy.Publisher(self.topics_names["brt"], Float32MultiArray, queue_size=10)
        self.constraints_map_pub = rospy.Publisher(self.topics_names["constraints_grid_map"], OccupancyGrid, queue_size=10)
        self.value_function_pub = rospy.Publisher(self.topics_names["value_function_at_state"], PoseStamped, queue_size=10)
        self.failure_pub = rospy.Publisher(self.topics_names["failure_set_at_state"], PoseStamped, queue_size=10)
        self.brt_computation_time_pub = rospy.Publisher(self.topics_names["brt_computation_time"], Float32, queue_size=10)
        self.brt_viz_pub = rospy.Publisher(self.topics_names["viz_brt"], OccupancyGrid, queue_size=10)

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
    
    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.robot_pose = np.array([pos.x, pos.y, euler[2]])

    # def odom_callback(self, msg: Odometry):
    #     pos = msg.pose.pose.position
    #     quat = msg.pose.pose.orientation
    #     euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    #     self.robot_pose = np.array([pos.x, pos.y, euler[2]])

    def floorplan_callback(self, msg: OccupancyGrid):
        # TODO: take initial orientation into account. For now will assume it is always 0
        self.map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.map_resolution = msg.info.resolution
        self.grid_map = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.reachability_solver is None:
            self._init_reachability_solver()

    def semantic_map_callback(self, msg: OccupancyGrid):
        self.semantic_grid_map = np.reshape(msg.data, (msg.info.height, msg.info.width))
        constraints_map = self.merge_maps()
        msg = self._construct_occupancy_grid_msg(constraints_map)
        self.constraints_map_pub.publish(msg)

    def compute_brt(self):
        if self.reachability_solver is None:
            rospy.logwarn("Reachability solver was not initialized yet. Cannot compute BRT.")
            return

        constraints_grid_map = self.merge_maps()
        constraints_grid_map = 1 - np.abs(constraints_grid_map)     # make -1 = occupied, 0 = occupied, 1 = free

        need_computation = True
        if not self.brt_computed and self.use_precomputed_brt:
            try:
                print("Trying to use precomputed brt...")
                self.values = np.load(self.possible_brt_file)
                print("Precomputed brt loaded")
                need_computation = False
                end_time = rospy.Time.now()
            except:
                print("failed to load precomputed brt..., move on to compute it")

        if need_computation:
            start_time = rospy.Time.now()
            self.values = self.reachability_solver.solve(constraints_grid_map, epsilon=self.epsilon)
            end_time = rospy.Time.now()
            time_taken = (end_time - start_time).to_sec()
            msg = Float32()
            msg.data = time_taken
            self.brt_computation_time_pub.publish(msg)
            if not self.brt_computed:
                print(f"save newly floorplan's brt to {self.possible_brt_file}")
                np.save(self.possible_brt_file, self.values)

        self.brt_computed = True
        self.last_updated = time.time()
    
    def publish_brt(self):
        if self.values is not None:
            msg = Float32MultiArray()
            (N, M, K) = self.values.shape
            flattened = self.values.flatten()

            # Set the flattened array as the data
            msg.data = flattened.tolist()

            # Set up the dimensions
            dim = MultiArrayDimension()
            dim.label = "N"
            dim.size = N
            dim.stride = N*M*K
            msg.layout.dim.append(dim)

            dim = MultiArrayDimension()
            dim.label = "M"
            dim.size = M
            dim.stride = M*K
            msg.layout.dim.append(dim)

            dim = MultiArrayDimension()
            dim.label = "K"
            dim.size = K
            dim.stride = K
            msg.layout.dim.append(dim)

            self.brt_pub.publish(msg)

    def merge_maps(self):
        """
        merges rtabmap and owl-vit constraints occupancy maps
        """
        if self.semantic_grid_map is None:
            rospy.logwarn("Semantic occupancy map was not initialized yet. Ignoring any language constraints..")
            return self.grid_map
        
        merged_map = self.grid_map.astype(int) + self.semantic_grid_map.astype(int)
        merged_map = np.clip(merged_map, a_min=-1, a_max=100, dtype=int)
        return merged_map

    def _init_reachability_solver(self):
        if self.grid_map is None:
            rospy.logwarn("Map was not received. Cannot initialize solver.")
            return
        
        size_y = self.grid_map.shape[0] * self.map_resolution
        size_x = self.grid_map.shape[1] * self.map_resolution
        domain_low = self.map_origin
        domain_high = domain_low + np.array([size_x, size_y])
        if self.use_precomputed_brt:
            brt_file = self.possible_brt_file
        else:
            brt_file = None
        self.reachability_solver = reachability.ReachabilitySolver(system="unicycle3d", 
                                                                    domain=[[domain_low[1], domain_low[0]],[domain_high[1], domain_high[0]]],
                                                                    unsafe_level=self.unsafe_level,
                                                                    vmin=self.vmin, vmax=self.vmax, wmax=self.wmax,
                                                                    possible_brt_file=brt_file,
                                                                    mode="brt", accuracy="low")

    def _construct_occupancy_grid_msg(self, map_data: np.array):
        msg = OccupancyGrid()
        origin = Pose()
        origin.position.x = self.map_origin[0]
        origin.position.y = self.map_origin[1]
        msg.info.origin = origin
        msg.info.resolution = self.map_resolution
        msg.info.height = map_data.shape[0]
        msg.info.width = map_data.shape[1]
        msg.data = map_data.flatten()
        return msg
    
    def _publish_brt_viz(self):
        if self.brt_computed and self.robot_pose is not None:
            print("publishing visualzation")
            msg = OccupancyGrid()
            msg.header.frame_id = 'map'
            msg.info.origin.position.x = self.map_origin[0]
            msg.info.origin.position.y = self.map_origin[1]
            msg.info.origin.orientation.w = 1
            msg.info.height = np.shape(self.grid_map)[0]
            msg.info.width = np.shape(self.grid_map)[1]
            msg.info.resolution = self.map_resolution
            ori = np.pi/2 - self.robot_pose[2]
            ori_idx = int(ori/(2*np.pi)*np.shape(self.values)[2])
            data = self.values[:, :, ori_idx]
            data = data.flatten()
            data[data < 0] = 0
            data[data > 0] = 100
            data = 100 - data
            msg.data = tuple(data.astype(int))
            self.brt_viz_pub.publish(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command Node")
    parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
    parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
    args = parser.parse_args()

    assert args.exp_path is not None, "a experiment config file must be provided"
    assert args.topics_path is not None, "topics names json file must be provided"

    rospy.init_node("brt_solver_node")
    # floorplan = np.load("/home/leo/git/hj_reachability/top_down_map.npy")
    solver_node = BRTSolverNode(args)

    rospy.sleep(5)
    
    # enforce first BRT computation
    while not solver_node.brt_computed and not rospy.is_shutdown():
        solver_node.compute_brt()
        solver_node.publish_brt()
        solver_node._publish_brt_viz()


    def compute_brt_thread(solver_node):
        while not rospy.is_shutdown():
            print(f'brt threading in progress')
            # time_interval = rospy.Time.now().secs - solver_node.last_updated.to_sec()
            time_interval = time.time() - solver_node.last_updated
            print(f'time since last brt: {time_interval}')
            # if time_interval >= 0.1:
            print(f"brt updating")
            solver_node.compute_brt()

    rate = rospy.Rate(10)
    computation_thread = threading.Thread(target=compute_brt_thread, args=(solver_node,))
    computation_thread.start()
    while not rospy.is_shutdown():
        solver_node._publish_brt_viz()
        solver_node.publish_brt()
        rate.sleep()