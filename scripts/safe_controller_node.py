# !/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid

import numpy as np

from lang_reachability import reachability


class SafeControllerNode:
    def __init__(self, floorplan) -> None:
        self.reachability_solver = reachability.ReachabilitySolver(system="unicycle3d", domain=[[-10, -10],[10, 10]], mode="brt", accuracy="low")
        self.grid_map = floorplan
        self.last_updated = -1

        self.nominal_action_sub = rospy.Subscriber("nominal_cmd_vel", Twist, queue_size=1, callback=self.nominal_action_callback)
        self.map_sub = rospy.Subscriber("grid_map", OccupancyGrid, callback=self.map_callback)

        self.safe_action_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        rospy.loginfo("Computing initial BRT. This may (it probably will) take a while.")
        self.values = self.compute_brt()

    def nominal_action_callback(self, msg: Twist):
        nominal_action = np.array([msg.linear.x, msg.angular.z])
        safe_action = self.get_safe_control(nominal_action, self.values)
        safe_action_msg = self._construct_twist_msg(safe_action)
        self.safe_action_pub.publish(safe_action_msg)

    def map_callback(self, msg: OccupancyGrid):
        self.grid_map = np.reshape(msg.data, (msg.info.height, msg.info.width))

    def compute_brt(self):
        if self.grid_map is None:
            rospy.logerr("Cannot compute BRT because map was not received")
            return
        
        rospy.loginfo("Computing BRT...")
        self.values = self.reachability_solver.solve(self.grid_map)
        self.last_updated = rospy.Time.now().secs
        rospy.loginfo("Finished BRT computation")

    def get_safe_control(self, nominal_action, values):
        # TODO
        # 1. check if nominal_action violates safe brt
        # 2.1. return nominal action if it does not
        # 2.2. obtain and return safe action if it does
        return nominal_action

    def _construct_twist_msg(self, action):
        """
        construct geometry_msgs/Twist assuming action = [v, w]
        """
        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        return msg


rospy.init_node("safe_controller_node")
floorplan = np.load("/home/leo/git/hj_reachability/top_down_map.npy")
solver_node = SafeControllerNode(floorplan)

brt_update_interval = 3
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    if rospy.Time.now().secs - solver_node.last_updated >= brt_update_interval:
        solver_node.compute_brt()

    rate.sleep()