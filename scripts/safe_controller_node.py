# !/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32

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
        self.value_function_pub = rospy.Publisher("value_function_at_state", PoseStamped, queue_size=10)
        self.failure_pub = rospy.Publisher("failure_set_value", PoseStamped, queue_size=10)
        self.safe_planning_time_pub = rospy.Publisher("safe_planning_time", Float32, queue_size=10)
        self.brt_computation_time_pub = rospy.Publisher("brt_computation_time", Float32, queue_size=10)

    def robot_pose_callback(self, msg: PoseWithCovarianceStamped):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.robot_pose = np.array([pos.x, pos.y, euler[2]])

    def nominal_action_callback(self, msg: Twist):
        if self.brt_computed:
            nominal_action = np.array([msg.linear.x, msg.angular.z])
            safe_action, value, initial_value = self.compute_safe_control(nominal_action)
            safe_action_msg = self._construct_twist_msg(safe_action)
            self.safe_action_pub.publish(safe_action_msg)
            value_function_msg = self._construct_pose_stamped_msg([self.robot_pose[0], self.robot_pose[1], value])
            self.value_function_pub.publish(value_function_msg)
            failure_msg = self._construct_pose_stamped_msg([self.robot_pose[0], self.robot_pose[1], initial_value])
            self.failure_pub.publish(failure_msg)
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
        
        start_time = rospy.Time.now().secs
        self.values = self.reachability_solver.solve(constraints_grid_map)
        end_time = rospy.Time.now().secs
        time_taken = end_time - start_time
        self.brt_computation_time_pub.publish(Float32(data=time_taken))

        self.brt_computed = True
        self.last_updated = end_time
        rospy.loginfo("Finished BRT computation")

    def compute_safe_control(self, nominal_action):
        if self.robot_pose is None:
            rospy.logerr("Robot pose was not received. Cannot compute safe action.")
            return nominal_action
        
        start_time = rospy.Time.now().secs
        safe_action, value = self.reachability_solver.compute_safe_control(self.robot_pose, nominal_action)
        time_taken = rospy.Time.now().secs - start_time
        self.safe_planning_time_pub.publish(Float32(data=time_taken))
        return safe_action, value
    
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
    
    def _construct_pose_stamped_msg(self, pos, ori):
        """
        construct geometry_msgs/Pose assuming pos = [x, y, z], ori = [roll, pitch, yaw]
        
        TODO: include orientation
        """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        
        msg.pose.orientation.w = 1.0
        return msg


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