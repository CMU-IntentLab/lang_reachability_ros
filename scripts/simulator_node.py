#!/usr/bin/env python3
import os, sys, json
from pathlib import Path

from lang_reachability import simulator as sim
from lang_reachability import systems

import rospy
import cv_bridge
import tf2_ros
import tf.transformations
import numpy as np
import logging
import argparse

from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion, TransformStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry, OccupancyGrid

class SimulatorNode:
    def __init__(self, args, dt=0.01) -> None:
        self.args = args
        self.exp_config = self.make_exp_config()
        self.data_root = self.exp_config['data_root']
        self.dataset_name = self.exp_config['dataset_name']
        self.scene_name = self.exp_config['scene_name']
        self.scene_path = self.make_scene_path()
        self.init_x = self.exp_config['initial_pose'][0]
        self.init_y = self.exp_config['initial_pose'][1]
        self.init_theta = self.exp_config['initial_pose'][2]
        self.sim = sim.Simulator(dataset_name=self.dataset_name, test_scene=self.scene_path, test_scene_name=self.scene_name,
                                 initial_state=[self.init_x, self.init_y, self.init_theta], dt=self.exp_config['dt'])
        # self.robot = systems.Unicycle3D(dt=dt)
        self.bridge = cv_bridge.CvBridge()
        self.robot_pose_pub = rospy.Publisher("gt_pose", Pose, queue_size=10)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10)
        self.robot_view_rgb_pub = rospy.Publisher("rgb/image", Image, queue_size=10)
        self.robot_view_depth_pub = rospy.Publisher("depth/image", Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher("rgb/camera_info", CameraInfo, queue_size=10)
        self.map_pub = rospy.Publisher("floor_plan", OccupancyGrid, queue_size=10)
        self.global_pose_pub = rospy.Publisher("global_pose", PoseWithCovarianceStamped, queue_size=10)

        self.cmd_vel_sub = rospy.Subscriber("cmd_vel", Twist, callback=self.cmd_vel_callback)
        self.goal_sub = rospy.Subscriber("goal", PoseStamped, callback=self.goal_callback)

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        self.vel = [0.0, 0.0]
        print('simulator node initialized')

    def make_exp_config(self):
        self.exp_path = self.args.exp_path
        with open(self.exp_path, 'r') as f:
            exp_config = json.load(f)
        return exp_config

    def goal_callback(self, msg: PoseStamped):
        # print(f"goal received: x={msg.pose.position.x}")
        pass

    def make_scene_path(self):
        scene_path = None
        if self.dataset_name == 'hssd':
            scene_path = os.path.join(self.data_root, "hssd", f"{self.scene_name}.glb")
        elif self.dataset_name == 'hssd-hab':
            scene_path = self.scene_name
        elif self.dataset_name == 'hm3d':
            sub_title = self.scene_name.split('-')[-1]
            scene_path = os.path.join(self.data_root, "hm3d/train", f"{self.scene_name}/{sub_title}.basis.glb")
        return scene_path

    def cmd_vel_callback(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z
        self.vel = [v, w]
        self.sim.agent_step(v, w)

    def update_sim(self):
        img = self.sim.get_rgb_observation()
        depth = self.sim.get_depth_observation()
        # topdown_view = sim.get_topdown_view()

        self.publish_floorplan()
        self.publish_robot_odom()
        self.publish_rgb_img(img)
        self.publish_depth_img(depth)
        self.publish_camera_info()
        # self.publish_global_pose()
        self._cameralink_to_baselink_tf()
        # self._map_to_odom_tf()

        self._odom_to_baselink_tf(*self.sim.get_agent_state_odom())

    def publish_floorplan(self):
        floorplan = OccupancyGrid()
        floorplan_data = np.flipud(self.sim.top_down_map)

        floorplan.header = Header()
        floorplan.header.stamp = rospy.Time.now()
        floorplan.header.frame_id = "map"

        floorplan.info.resolution = self.sim.map_resolution
        floorplan.info.width = floorplan_data.shape[1]
        floorplan.info.height = floorplan_data.shape[0]
        floorplan.info.origin.position.x = -floorplan.info.width * floorplan.info.resolution / 2
        floorplan.info.origin.position.y = -floorplan.info.height * floorplan.info.resolution / 2
        floorplan.info.origin.position.z = 0.0
        floorplan.info.origin.orientation.x = 0.0
        floorplan.info.origin.orientation.y = 0.0
        floorplan.info.origin.orientation.z = 0.0
        floorplan.info.origin.orientation.w = 1.0

        floorplan.data = floorplan_data.flatten().tolist()

        self.map_pub.publish(floorplan)

    def publish_camera_info(self):
        K = self.sim.get_camera_intrinsics_mat()
        caminfo_msg = self._construct_camera_info_message(K)
        self.camera_info_pub.publish(caminfo_msg)

    def publish_rgb_img(self, img):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"
        rgb_msg = self.bridge.cv2_to_imgmsg(img, encoding="rgba8", header=header)
        self.robot_view_rgb_pub.publish(rgb_msg)

    def publish_depth_img(self, depth):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"
        # depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1", header=header)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough", header=header)
        self.robot_view_depth_pub.publish(depth_msg)

    def publish_global_pose(self):
        pose = PoseWithCovarianceStamped()

        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        x, y, theta = self.sim.robot_state_world

        pose.pose.pose.position.x = x
        pose.pose.pose.position.y = y
        pose.pose.pose.position.z = 0.0  # Assuming 2D plane

        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        pose.pose.pose.orientation = Quaternion(*quat)

        # Set covariance (assuming zero covariance for simplicity, adjust as needed)
        pose.pose.covariance = [0.0] * 36

        self.global_pose_pub.publish(pose)

    def publish_robot_odom(self):
        state = self.sim.robot.state
        odom_msg = self._construct_odom_message(*self.sim.get_agent_state_odom())
        self.odom_pub.publish(odom_msg)

    def _construct_odom_message(self, x, y, theta):
        odom = Odometry()
        
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0  # Assuming a 2D plane
        
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        odom.pose.pose.orientation = Quaternion(*quat)
        
        odom.pose.covariance = [0]*36
        odom.twist.covariance = [0]*36
        
        odom.twist.twist.linear.x = self.vel[0]
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self.vel[1]

        self.vel = [0.0, 0.0]
        
        return odom
    
    def _construct_camera_info_message(self, K):
        # Create a CameraInfo message
        camera_info = CameraInfo()

        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = "camera_link"
        
        # Set the intrinsic camera matrix (K) - must be a 3x3 matrix
        camera_info.K = K.flatten().tolist()
        
        # Set the distortion coefficients (D) - assuming no distortion
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Set the camera matrix (P) - projection matrix
        camera_info.P = [0.0] * 12
        camera_info.P[0:3] = K[0, 0:3]
        camera_info.P[4:7] = K[1, 0:3]
        camera_info.P[8:11] = K[2, 0:3]
        
        # Set the rectification matrix (R) - assuming identity (no rectification)
        camera_info.R = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        
        # Set other fields as necessary (width, height, etc.)
        camera_info.width = 256   # Example width
        camera_info.height = 256  # Example height
        
        return camera_info

    def _map_to_odom_tf(self):
        static_transform = TransformStamped()

        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "map"
        static_transform.child_frame_id = "odom"

        static_transform.transform.translation.x = self.init_x
        static_transform.transform.translation.y = self.init_y
        static_transform.transform.translation.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0, 0, self.init_theta)

        static_transform.transform.rotation = Quaternion(*quat)

        self.broadcaster.sendTransform(static_transform)

    def _cameralink_to_baselink_tf(self):
        static_transform = TransformStamped()

        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "base_link"
        static_transform.child_frame_id = "camera_link"

        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 1.0

        quat = tf.transformations.quaternion_from_euler(-1.57, 0, -1.57)

        static_transform.transform.rotation = Quaternion(*quat)

        self.broadcaster.sendTransform(static_transform)

    def _odom_to_baselink_tf(self, x, y, theta):
        transform = TransformStamped()

        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0, 0, theta)

        transform.transform.rotation = Quaternion(*quat)

        self.broadcaster.sendTransform(transform)

    def setup_logger(self, log_file):
        dir_path = str(Path(__file__).parent.parent)
        logs_dir = os.path.join(dir_path, 'logs')
        log_file_path = os.path.join(logs_dir, log_file)
        Path(logs_dir).mkdir(exist_ok=True)

        logger = logging.getLogger('command_node')
        logger.setLevel(logging.INFO)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # Capture logs from external libraries
        logging.getLogger('habitat_sim').setLevel(logging.INFO)
        logging.getLogger('habitat_sim').addHandler(fh)
        logging.getLogger('habitat_sim').addHandler(ch)

        return logger


if __name__ == '__main__':
    rospy.init_node("simulator")
    rate = rospy.Rate(10)

    parser = argparse.ArgumentParser(description="Simulator Node")
    parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
    parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
    args = parser.parse_args()

    sim_node = SimulatorNode(args=args)

    while not rospy.is_shutdown():
        sim_node.update_sim()
        rate.sleep()