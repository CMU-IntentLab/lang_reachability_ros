# !/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import tf.transformations as tft 

import numpy as np

from lang_reachability import reachability


class SafeControllerNode:
    def __init__(self) -> None:
        self.reachability_solver = None
        self.robot_pose = None
        self.map_resolution = -1
        self.map_origin = None
        self.grid_map = None
        self.semantic_grid_map = None
        self.last_updated = -1
        self.brt_computed = False

        self.robot_pose_sub = rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped, callback=self.robot_pose_callback)
        self.nominal_action_sub = rospy.Subscriber("nominal_cmd_vel", Twist, queue_size=1, callback=self.nominal_action_callback)
        self.floorplan_sub = rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid, callback=self.floorplan_callback)
        self.semantic_map_sub = rospy.Subscriber("semantic_grid_map", OccupancyGrid, callback=self.semantic_map_callback)

        self.safe_action_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.robot_pose = np.array([pos.x, pos.y, euler[2]])

    def nominal_action_callback(self, msg: Twist):
        if self.brt_computed:
            nominal_action = np.array([msg.linear.x, msg.angular.z])
            safe_action = self.compute_safe_control(nominal_action)
            safe_action_msg = self._construct_twist_msg(safe_action)
            self.safe_action_pub.publish(safe_action_msg)
        else:
            rospy.logwarn("BRT was not computed yet. Not protecting against nominal action!")
            nominal_action = np.array([msg.linear.x, msg.angular.z])
            nominal_action_msg = self._construct_twist_msg(nominal_action)
            self.safe_action_pub.publish(nominal_action_msg)

    def floorplan_callback(self, msg: OccupancyGrid):
        # TODO: take initial orientation into account. For now will assume it is always 0
        self.map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.map_resolution = msg.info.resolution
        self.grid_map = np.reshape(msg.data, (msg.info.height, msg.info.width))
        if self.reachability_solver is None:
            self._init_reachability_solver()

    def semantic_map_callback(self, msg: OccupancyGrid):
        self.semantic_grid_map = np.reshape(msg.data, (msg.info.height, msg.info.width))

    def compute_brt(self):
        if self.reachability_solver is None:
            rospy.logerr("Reachability solver was not initialized yet. Cannot compute BRT.")
            return
        
        if not self.brt_computed:
            rospy.loginfo("Computing initial BRT. This may (it probably will) take a while.")
        else:
            rospy.loginfo("Computing warm-started BRT.")

        constraints_grid_map = self.merge_maps()
        constraints_grid_map = 1 - constraints_grid_map     # make 0 = occupied, 1 = free
        rospy.loginfo(f"grid map shape = {np.shape(constraints_grid_map)}")
        self.values = self.reachability_solver.solve(constraints_grid_map)
        self.last_updated = rospy.Time.now().secs
        rospy.loginfo("Finished BRT computation")
        self.brt_computed = True

    def compute_safe_control(self, nominal_action):
        if self.robot_pose is None:
            rospy.logerr("Robot pose was not received. Cannot compute safe action.")
            return nominal_action
        
        safe_action = self.reachability_solver.compute_safe_control(self.robot_pose, nominal_action)
        return safe_action
    
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
            rospy.logerr("Map was not received. Cannot initialize solver.")
            return
        
        size_y = self.grid_map.shape[0] * self.map_resolution
        size_x = self.grid_map.shape[1] * self.map_resolution
        domain_low = self.map_origin
        domain_high = domain_low + np.array([size_x, size_y])
        rospy.loginfo(f"domain = {domain_low, domain_high}")
        self.reachability_solver = reachability.ReachabilitySolver(system="unicycle3d", 
                                                                domain=[[domain_low[0], domain_low[1]],[domain_high[0], domain_high[1]]], 
                                                                mode="brt", accuracy="low")

    def _construct_twist_msg(self, action):
        """
        construct geometry_msgs/Twist assuming action = [v, w]
        """
        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        return msg
    
    def __position_to_cell(self, x, y, origin='lower'):
        """
        converts (x, y) position to (row, column) cell in the grid map
        """
        column = (x - self.map_origin[0])/self.map_resolution
        row = (y - self.map_origin[1])/self.map_resolution
        if origin == 'upper':
            row = self.semantic_grid_map.shape[0] - y
        return row.astype(int), column.astype(int)


rospy.init_node("safe_controller_node")
# floorplan = np.load("/home/leo/git/hj_reachability/top_down_map.npy")
solver_node = SafeControllerNode()

# enforce first BRT computation
while not solver_node.brt_computed and not rospy.is_shutdown():
    solver_node.compute_brt()
    rospy.sleep(2)


brt_update_interval = 15
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    if rospy.Time.now().secs - solver_node.last_updated >= brt_update_interval:
        solver_node.compute_brt()

    rate.sleep()