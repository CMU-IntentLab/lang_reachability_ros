# !/usr/bin/env python
import argparse
import json

import rospy

from geometry_msgs.msg import Twist, TwistStamped, PoseWithCovarianceStamped, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32, Float32MultiArray

import tf2_ros
import tf.transformations as tft 

import numpy as np

from lang_reachability import reachability


class SafeControllerNode:
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
        self.values = None
        self.brt_computed = False

        self.vmin = self.exp_config["vmin"]
        self.vmax = self.exp_config["vmax"]
        self.wmax = self.exp_config["wmax"]
        self.brt_update_interval = self.exp_config["brt_update_interval"]
        self.epsilon = self.exp_config["brt_convergence_epsilon"]
        self.unsafe_level = self.exp_config["brt_unsafe_level"]

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.brt_sub = rospy.Subscriber(self.topics_names["brt"], Float32MultiArray, callback=self.brt_callback)
        self.robot_pose_sub = rospy.Subscriber(self.topics_names["pose"], PoseWithCovarianceStamped, callback=self.robot_pose_callback)
        # self.robot_odom_sub = rospy.Subscriber(self.topics_names["odom"], Odometry, callback=self.odom_callback)
        self.nominal_action_sub = rospy.Subscriber(self.topics_names["nominal_action"], Twist, queue_size=10, callback=self.nominal_action_callback)
        self.floorplan_sub = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, callback=self.floorplan_callback)
        self.semantic_map_sub = rospy.Subscriber(self.topics_names["semantic_grid_map"], OccupancyGrid, callback=self.semantic_map_callback)

        self.constraints_map_pub = rospy.Publisher(self.topics_names["constraints_grid_map"], OccupancyGrid, queue_size=10)
        self.safe_action_pub = rospy.Publisher(self.topics_names["safe_action"], Twist, queue_size=10)
        self.safe_action_viz_pub = rospy.Publisher("cmd_vel_viz", TwistStamped, queue_size=1)
        self.value_function_pub = rospy.Publisher(self.topics_names["value_function_at_state"], PoseStamped, queue_size=10)
        self.failure_pub = rospy.Publisher(self.topics_names["failure_set_at_state"], PoseStamped, queue_size=10)
        self.safe_planning_time_pub = rospy.Publisher(self.topics_names["safe_planning_time"], Float32, queue_size=10)
        self.brt_computation_time_pub = rospy.Publisher(self.topics_names["brt_computation_time"], Float32, queue_size=10)
        self.brt_viz_pub = rospy.Publisher('brt_viz', OccupancyGrid, queue_size=10)

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

    def brt_callback(self, msg: Float32MultiArray):
        # Extract dimensions N, M, K from the layout information
        N = msg.layout.dim[0].size
        M = msg.layout.dim[1].size
        K = msg.layout.dim[2].size

        # Convert the data back into a numpy array with the extracted shape (N, M, K)
        self.values = np.array(msg.data).reshape((N, M, K))
        self.values_grad = np.gradient(self.values)
        self.brt_computed = True
        # rospy.loginfo("updated brt")

    def nominal_action_callback(self, msg: Twist):
        if self.brt_computed and self.reachability_solver is not None:
            # rospy.loginfo("Returning safe action")
            nominal_action = np.array([msg.linear.x, msg.angular.z])
            safe_action, value, initial_value = self.compute_safe_control(nominal_action)
            print(safe_action)
            safe_action_msg = self._construct_twist_msg(safe_action)
            self.safe_action_pub.publish(safe_action_msg)
            safe_action_viz_msg = self._construct_twist_stamped_msg(safe_action)
            self.safe_action_viz_pub.publish(safe_action_viz_msg)
            value_function_msg = self._construct_pose_stamped_msg([self.robot_pose[0], self.robot_pose[1], value], [0.0, 0.0, self.robot_pose[2]])
            self.value_function_pub.publish(value_function_msg)
            # failure_msg = self._construct_pose_stamped_msg([self.robot_pose[0], self.robot_pose[1], initial_value], [0.0, 0.0, self.robot_pose[2]])
            # self.failure_pub.publish(failure_msg)
        # else:
        #     # rospy.loginfo("Returning nominal action")
        #     nominal_action = np.array([0.0, 0.0])
        #     nominal_action_msg = self._construct_twist_msg(nominal_action)
        #     self.safe_action_pub.publish(nominal_action_msg)

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
        rospy.loginfo(f"grid map shape = {np.shape(constraints_grid_map)}")

        if not self.brt_computed:
            rospy.loginfo("Computing initial BRT. This may (it probably will) take a while.")
        else:
            rospy.loginfo("Computing warm-started BRT.")

        start_time = rospy.Time.now()
        self.values = self.reachability_solver.solve(constraints_grid_map)
        end_time = rospy.Time.now()
        time_taken = (end_time - start_time).to_sec()
        self.brt_computation_time_pub.publish(Float32(data=time_taken))

        self.brt_computed = True
        self.last_updated = end_time
        rospy.loginfo("Finished BRT computation")

    def compute_safe_control(self, nominal_action):
        if self.robot_pose is None:
            rospy.logwarn("Robot pose was not received. Cannot compute safe action.")
            return nominal_action
        
        start_time = rospy.Time.now().to_sec()
        safe_action, value, initial_values = self.reachability_solver.compute_safe_control(self.robot_pose, nominal_action, self.values, self.values_grad)
        time_taken = rospy.Time.now().to_sec() - start_time
        self.safe_planning_time_pub.publish(Float32(data=time_taken))
        return safe_action, value, initial_values
    
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
        converged_values = np.load("/home/leo/riss_ws/src/lang_reachability_ros/lab_value_function.npy")
        self.reachability_solver = reachability.ReachabilitySolver(system="unicycle3d", 
                                                                    domain=[[domain_low[1], domain_low[0]],[domain_high[1], domain_high[0]]],
                                                                    unsafe_level=self.unsafe_level,
                                                                    vmin=self.vmin, vmax=self.vmax, wmax=self.wmax,
                                                                    converged_values=None,
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

    def _construct_twist_msg(self, action):
        """
        construct geometry_msgs/Twist assuming action = [v, w]
        """
        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        return msg
    
    def _construct_twist_stamped_msg(self, action):
        """
        construct geometry_msgs/Twist assuming action = [v, w]
        """
        msg = TwistStamped()
        msg.header.frame_id = "locobot/base_link"
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = action[0]
        msg.twist.angular.z = action[1]
        return msg
    
    def _construct_pose_stamped_msg(self, pos, ori):
        """
        construct geometry_msgs/Pose assuming pos = [x, y, z], ori = [roll, pitch, yaw]
        """
        quat = tft.quaternion_from_euler(ai=ori[0], aj=ori[1], ak=ori[2])
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        return msg
    
    # def _publish_brt_viz(self):
    #     if self.brt_computed:
    #         msg = OccupancyGrid()
    #         msg.header.frame_id = 'map'
    #         msg.info.origin.position.x = self.map_origin[0]
    #         msg.info.origin.position.y = self.map_origin[1]
    #         msg.info.origin.orientation.w = 1
    #         msg.info.height = np.shape(self.grid_map)[0]
    #         msg.info.width = np.shape(self.grid_map)[1]
    #         msg.info.resolution = self.map_resolution
    #         ori = np.pi/2 - self.robot_pose[2]
    #         ori_idx = int(ori/(2*np.pi)*np.shape(self.values)[2])
    #         data = self.values[:, :, ori_idx]
    #         data = data.flatten()
    #         data[data < 0] = 0
    #         data[data > 0] = 100
    #         data = 100 - data
    #         msg.data = tuple(data.astype(int))
    #         self.brt_viz_pub.publish(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command Node")
    parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
    parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
    args = parser.parse_args()

    assert args.exp_path is not None, "a experiment config file must be provided"
    assert args.topics_path is not None, "topics names json file must be provided"

    rospy.init_node("safe_controller_node")
    # floorplan = np.load("/home/leo/git/hj_reachability/top_down_map.npy")
    solver_node = SafeControllerNode(args)

    rospy.sleep(5)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()