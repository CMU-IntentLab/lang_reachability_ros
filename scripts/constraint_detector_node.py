#!/usr/bin/env python3

from lang_reachability import perception

import rospy
import cv_bridge
import tf2_ros
import json
import tf.transformations
import argparse

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
import cv2


class ConstraintDetectorNode:
    def __init__(self, args) -> None:
        self.args = args
        self.exp_config = self.make_exp_config()
        self.topics_names = self.make_topics_names()
        self.rgb_img = None
        self.depth_img = None
        self.robot_pose = None

        # camera params
        self.tf_camera_link_frame = self.exp_config["tf_camera_link_frame"]
        self.K = None
        self.K_inv = None
        self.T = None
        self.T_inv = None

        # map params
        self.tf_map_frame = self.exp_config["tf_map_frame"]
        self.resolution = None
        self.origin = None
        self.semantic_grid_map = None
        self.grid_map = None

        self.object_detector = perception.ObjectDetector(score_threshold=self.exp_config['score_threshold'], init_queries=self.exp_config['text_queries'])
        self.bridge = cv_bridge.CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.vlm_detections_pub = rospy.Publisher(self.topics_names["vlm_detections"], Image, queue_size=10)
        self.constaints_grid_map_pub = rospy.Publisher(self.topics_names["constraints_grid_map"], OccupancyGrid, queue_size=10)
        self.semantic_grid_map_pub = rospy.Publisher(self.topics_names["semantic_grid_map"], OccupancyGrid, queue_size=10)

        self.text_query_sub = rospy.Subscriber(self.topics_names["language_constraint"], String, callback=self.text_query_callback)
        self.grid_map_sub = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, callback=self.grid_map_callback)
        self.rgb_img_sub = rospy.Subscriber(self.topics_names["rgb_image"], Image, callback=self.rgb_img_callback)
        self.depth_img_sub = rospy.Subscriber(self.topics_names["depth_image"], Image, callback=self.depth_img_callback)
        self.camera_info_sub = rospy.Subscriber(self.topics_names["camera_info"], CameraInfo, callback=self.camera_info_callback)
        self.robot_pose_sub = rospy.Subscriber(self.topics_names["pose"], PoseWithCovarianceStamped, callback=self.robot_pose_callback)

        rospy.loginfo(f"Initialized VLM with the following cofigs: [ \
                        language constraints: {self.object_detector.text_queries} \n \
                        score threshold: {self.object_detector.score_threshold} \
                        ]")

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

    def text_query_callback(self, msg: String):
        self.object_detector.add_new_text_query(msg.data)
        rospy.loginfo(f"Added '{msg.data}' to the list of language constraints. Current constraints are '{self.object_detector.text_queries}'.")

    def grid_map_callback(self, msg: OccupancyGrid):
        self.resolution = msg.info.resolution
        self.origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y]) # assumes orientation is always 0
        self.grid_map = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.semantic_grid_map is None:
            self.__init_semantic_grid_map()
        elif self.semantic_grid_map.shape != self.grid_map.shape:
            self.__update_semantic_grid_map_info()

    def rgb_img_callback(self, msg: Image):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def depth_img_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") #/1000 uncomment if using the realsense camera

    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None or self.K_inv is None:
            self.K = np.reshape(msg.K, (3, 3))
            self.K_inv = np.linalg.inv(self.K)

    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.robot_pose = np.array([pos.x, pos.y, euler[2]])

    def update_constraints_map(self):
        if self.rgb_img is None or self.depth_img is None or self.robot_pose is None:
            print(f"vlm status: rgb_img={self.rgb_img is not None}, depth_img={self.depth_img is not None}, robot_pose={self.robot_pose is not None}")
            return
        
        rgb_img = np.copy(self.rgb_img)
        detections = self.object_detector.detect(rgb_img)
        K_inv = self.get_inv_camera_intrinsics_matrix()
        T_inv = self.get_inv_camera_extrinsics_matrix()
        # print(f"T_inv={T_inv}")
        for bbox, label in detections:
            x_occ, y_occ = self.object_detector.estimate_object_position(self.depth_img, bbox, K_inv, T_inv, threshold=4)
            row, col = self.__position_to_cell(x_occ, y_occ)
            try:
                self.semantic_grid_map[row, col] = 100
            except IndexError:
                continue

            cv2.rectangle(rgb_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        self.publish_vlm_detections(rgb_img)
        self.publish_semantic_grid_map()
        # merged_map = self.merge_maps()
        # if merged_map is not None:
        #     self.publish_constraint_grid_map(merged_map)

    # def merge_maps(self):
    #     """
    #     merges rtabmap and owl-vit constraints occupancy maps
    #     """
    #     if self.semantic_grid_map is None:
    #         rospy.logwarn("Cannot merge maps because semantic occupancy map has not been initialized yet.")
    #         return None
        
    #     merged_map = self.grid_map.astype(int) + self.semantic_grid_map.astype(int)
    #     merged_map = np.clip(merged_map, a_min=-1, a_max=100, dtype=int)
    #     return merged_map

    def publish_vlm_detections(self, rgb_img):
        msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        self.vlm_detections_pub.publish(msg)

    def publish_semantic_grid_map(self):
        if self.semantic_grid_map is not None:
            msg = self.__construct_map_msg(self.semantic_grid_map)
            self.semantic_grid_map_pub.publish(msg)

    def publish_constraint_grid_map(self, merged_map):
        """
        publishes joint rtabmap + owl-vit constraints occupancy map
        """
        msg = self.__construct_map_msg(merged_map)
        self.constaints_grid_map_pub.publish(msg)

    def get_camera_extrinsics_matrix(self):
        transform_camera_to_map = self.tf_buffer.lookup_transform(self.tf_camera_link_frame, self.tf_map_frame, rospy.Time(0), rospy.Duration(1.0))
        t = transform_camera_to_map.transform.translation
        q = transform_camera_to_map.transform.rotation

        matrix = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z

        return matrix
    
    def get_inv_camera_extrinsics_matrix(self):
        transform_camera_to_map = self.tf_buffer.lookup_transform(self.tf_map_frame, self.tf_camera_link_frame, rospy.Time(0), rospy.Duration(1.0))
        t = transform_camera_to_map.transform.translation
        q = transform_camera_to_map.transform.rotation

        matrix = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z

        return matrix

    def get_camera_intrinsics_matrix(self):
        return self.K
    
    def get_inv_camera_intrinsics_matrix(self):
        return self.K_inv
    
    def __init_semantic_grid_map(self):
        """
        initializes semantic constraints map as an empty copy of the slam sysyem's output
        """
        self.semantic_grid_map = np.zeros(self.grid_map.shape, dtype=int)
        rospy.loginfo("Initialized semantic map")

    def __update_semantic_grid_map_info(self):
        old_map = np.copy(self.semantic_grid_map)
        self.semantic_grid_map = np.zeros(self.grid_map.shape)
        row, col = self.__position_to_cell(self.origin[0], self.origin[1])
        row_max, row_min = row + old_map.shape[0], row
        col_max, col_min = col + old_map.shape[1], col
        self.semantic_grid_map[row_min:row_max, col_min:col_max] = old_map

    def __position_to_cell(self, x, y, origin='lower'):
        """
        converts (x, y) position to (row, column) cell in the grid map
        """
        column = (x - self.origin[0])/self.resolution
        row = (y - self.origin[1])/self.resolution
        if origin == 'upper':
            row = self.semantic_grid_map.shape[0] - y
        return row.astype(int), column.astype(int)

    def __construct_map_msg(self, grid_map: np.array):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.tf_map_frame
        msg.data = tuple(grid_map.flatten().astype(int))
        msg.info.resolution = self.resolution
        msg.info.height = grid_map.shape[0]
        msg.info.width = grid_map.shape[1]
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0
        return msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command Node")
    parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
    parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
    args = parser.parse_args()

    assert args.exp_path is not None, "a experiment config file must be provided"
    assert args.topics_path is not None, "topics names json file must be provided"

    rospy.init_node("constraint_detector_node")
    node = ConstraintDetectorNode(args)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        node.update_constraints_map()
        rate.sleep()
