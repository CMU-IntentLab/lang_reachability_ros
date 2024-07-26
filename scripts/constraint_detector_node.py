#!/usr/bin/env python3

import sys
# sys.path.append("/home/riss_ws/src/lang_reachability_ros/scripts/lang-reachability")

from lang_reachability import perception

import rospy
import cv_bridge
import tf2_ros
import tf.transformations

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo

import numpy as np
import cv2

# TODO:
# - save previously detected constraints. self.__init_semantic_map is being called every time we receive a new map from rtabmap!!

class ConstraintDetectorNode:
    def __init__(self) -> None:
        self.object_detector = perception.ObjectDetector(score_threshold=0.2)
        self.bridge = cv_bridge.CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.vlm_detections_pub = rospy.Publisher("vlm_detections", Image, queue_size=10)
        self.constaints_grid_map_pub = rospy.Publisher("constraints_grid_map", OccupancyGrid, queue_size=10)
        self.semantic_grid_map_pub = rospy.Publisher("semantic_grid_map", OccupancyGrid, queue_size=10)

        self.text_query_sub = rospy.Subscriber("language_constraint", String, callback=self.text_query_callback)
        self.grid_map_sub = rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid, callback=self.grid_map_callback)
        self.rgb_img_sub = rospy.Subscriber("rgb/image", Image, callback=self.rgb_img_callback)
        self.depth_img_sub = rospy.Subscriber("depth/image", Image, callback=self.depth_img_callback)
        self.camera_info_sub = rospy.Subscriber("rgb/camera_info", CameraInfo, callback=self.camera_info_callback)
        
        self.rgb_img = None
        self.depth_img = None

        # camera params
        self.K = None
        self.K_inv = None
        self.T = None
        self.T_inv = None

        # map params
        self.resolution = None
        self.origin = None
        self.semantic_grid_map = None
        self.grid_map = None

    def text_query_callback(self, msg: String):
        self.object_detector.add_new_text_query(msg.data)

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
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None or self.K_inv is None:
            self.K = np.reshape(msg.K, (3, 3))
            self.K_inv = np.linalg.inv(self.K)

    def update_constraints_map(self):
        if self.rgb_img is None or self.depth_img is None:
            return
        
        rgb_img = np.copy(self.rgb_img)
        detections = self.object_detector.detect(rgb_img)
        K_inv = self.get_inv_camera_intrinsics_matrix()
        T_inv = self.get_inv_camera_extrinsics_matrix()
        for bbox, label in detections:
            x_occ, y_occ = self.object_detector.estimate_object_position(self.depth_img, bbox, K_inv, T_inv)
            row, col = self.__position_to_cell(x_occ, y_occ)
            try:
                self.semantic_grid_map[row, col] = 100
            except IndexError:
                continue

            cv2.rectangle(rgb_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        self.publish_vlm_detections(rgb_img)
        self.publish_semantic_grid_map()
        merged_map = self.merge_maps()
        if merged_map is not None:
            self.publish_constraint_grid_map(merged_map)

    def merge_maps(self):
        """
        merges rtabmap and owl-vit constraints occupancy maps
        """
        if self.semantic_grid_map is None:
            rospy.logwarn("Cannot merge maps because semantic occupancy map has not been initialized yet.")
            return None
        
        merged_map = self.grid_map.astype(int) + self.semantic_grid_map.astype(int)
        merged_map = np.clip(merged_map, a_min=-1, a_max=100, dtype=int)
        return merged_map

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
        transform_camera_to_map = self.tf_buffer.lookup_transform('camera_link', 'map', rospy.Time(0), rospy.Duration(1.0))
        t = transform_camera_to_map.transform.translation
        q = transform_camera_to_map.transform.rotation

        matrix = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z

        return matrix
    
    def get_inv_camera_extrinsics_matrix(self):
        transform_camera_to_map = self.tf_buffer.lookup_transform('map', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
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
        msg.header.frame_id = "map"
        msg.data = tuple(grid_map.flatten().astype(int))
        msg.info.resolution = self.resolution
        msg.info.height = grid_map.shape[0]
        msg.info.width = grid_map.shape[1]
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0
        return msg

rospy.init_node("constraint_detector_node")
node = ConstraintDetectorNode()

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    node.update_constraints_map()
    rate.sleep()
