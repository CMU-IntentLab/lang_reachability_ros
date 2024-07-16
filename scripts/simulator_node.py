#!/usr/bin/env python3
import os, sys, json
from pathlib import Path

from lang_reachability import simulator as sim
from lang_reachability import systems

import rospy
import cv_bridge
import tf2_ros
import tf.transformations

from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry


class SimulatorNode:
    def __init__(self, dataset_name, test_scene, dt=0.01, init_x=0.0, init_y=2.5, init_theta=0.0) -> None:
        self.sim = sim.Simulator(dataset_name=dataset_name, test_scene=test_scene)
        self.robot = systems.Dubins3D(init_x=init_x, init_y=init_y, init_theta=init_theta, dt=dt)
        self.bridge = cv_bridge.CvBridge()

        self.robot_pose_pub = rospy.Publisher("robot/gt_pose", Pose, queue_size=10)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10)
        self.robot_view_rgb_pub = rospy.Publisher("rgb/image", Image, queue_size=10)
        self.robot_view_depth_pub = rospy.Publisher("depth/image", Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher("rgb/camera_info", CameraInfo, queue_size=10)
        
        self.cmd_vel_sub = rospy.Subscriber("cmd_vel", Twist, callback=self.cmd_vel_callback)

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        self.vel = [0.0, 0.0]

    def cmd_vel_callback(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z
        self.vel = [v, w]
        self.robot.step(v, w)

    def update_sim(self):
        state = self.robot.state
        img = self.sim.get_rgb_observation(state[0], state[1], state[2])
        depth = self.sim.get_depth_observation(state[0], state[1], state[2])
        # topdown_view = sim.get_topdown_view()

        self.publish_robot_odom()
        self.publish_rgb_img(img)
        self.publish_depth_img(depth)
        self.publish_camera_info()
        self._cameralink_to_baselink_tf()
        self._odom_to_baselink_tf(state[0], state[1], state[2])

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

    def publish_robot_odom(self):
        state = self.robot.state
        odom_msg = self._construct_odom_message(state[0], state[1], state[2])
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

def setup_data():
    dir_path = str(Path(__file__).parent.parent)
    with open(os.path.join(dir_path, 'scripts/lang_reachability/configs/path_config.json')) as path_config_file:
        path_config = json.load(path_config_file)

    data_root = path_config['data_root']
    dataset_name = 'hssd'
    scene_idx = 0
    if dataset_name == 'hssd':
        scene_map = {0: "102344469", 1: "102344022", 2: "102344094", 3: "103997403_171030405", 4: "102815859", 5:"102816216"}
        test_scene_name = scene_map[scene_idx]
        test_scene = os.path.join(data_root, "hssd", f"{test_scene_name}.glb")
    elif dataset_name == 'hssd-hab':
        scene_map = {0: "102344469", 1: "102344022", 2: "102344094", 3: "103997403_171030405", 4: "102815859",
                    5: "102816216", 6: "102344094_raw", 7: "102344094_mod"}
        test_scene_name = scene_map[scene_idx]
        test_scene = test_scene_name
    elif dataset_name == 'hm3d':
        scene_map = {0: "00099-226REUyJh2K", 1: "00013-sfbj7jspYWj", 2: "00198-eZrc5fLTCmi"}
        test_scene_name = scene_map[scene_idx]
        sub_title = test_scene_name.split('-')[-1]
        test_scene = os.path.join(data_root, "hm3d/train", f"{test_scene_name}/{sub_title}.basis.glb")
    return dataset_name, test_scene

rospy.init_node("simulator")
rate = rospy.Rate(10)

dataset_name, test_scene = setup_data()
sim_node = SimulatorNode(dataset_name=dataset_name, test_scene=test_scene)

while not rospy.is_shutdown():
    sim_node.update_sim()
    rate.sleep()