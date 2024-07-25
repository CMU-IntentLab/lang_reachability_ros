import rospy
import torch
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Twist
from visualization_msgs.msg import Marker, MarkerArray
from lang_reachability import navigator

class NavigationNode:
    def __init__(self):
        self.odom_sub = rospy.Subscriber("odom", Odometry, callback=self.odom_callback)
        self.goal_sub = rospy.Subscriber("goal", PoseStamped, callback=self.goal_callback)
        self.map_sub = rospy.Subscriber("floor_plan", OccupancyGrid, callback=self.map_callback)

        self.twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.marker_pub = rospy.Publisher("vis_trajectories", MarkerArray, queue_size=1)

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._dtype = torch.float32
        self._navigator = navigator.Navigator(planner_type="mppi", device=self._device, dtype=self._dtype, dt=0.1)
        self.navigator_ready = False

        self._odom = None
        self._goal = None
        self._map = None

        self._goal_thresh = 0.1

    def _check_navigator_ready(self):
        if self._odom is not None and self._goal is not None and self._map is not None:
            self.navigator_ready = True
            print(f'begin navigation: goal: ({self._goal.pose.position.x}, {self._goal.pose.position.y})')

    def odom_callback(self, msg: Odometry) -> None:
        self._odom = msg
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self._navigator.set_odom(position=pos, orientation=quat)
        if not self.navigator_ready:
            self._check_navigator_ready()

    def map_callback(self, msg: OccupancyGrid) -> None:
        self._map = msg
        map_data = msg.data
        map_dim = [msg.info.height, msg.info.width]
        map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        map_resolution = msg.info.resolution
        self._navigator.set_map(map_data=map_data, map_dim=map_dim, map_origin=map_origin, map_resolution=map_resolution)
        if not self.navigator_ready:
            self._check_navigator_ready()

    def goal_callback(self, msg: PoseStamped) -> None:
        self._goal = msg
        position = [msg.pose.position.x, msg.pose.position.y]
        quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        self._navigator.set_goal(position=position, orientation=quat)
        if not self.navigator_ready:
            self._check_navigator_ready()

    def _goal_achieved(self):
        x = self._odom.pose.pose.position.x
        y = self._odom.pose.pose.position.y

        dist_goal = np.sqrt((x - self._goal.pose.position.x) ** 2 + (y - self._goal.pose.position.y) ** 2)
        if dist_goal < self._goal_thresh:
            return True
        else:
            return False

    def publish_command(self):
        v, w = self._navigator.get_command()
        print(v, w)
        twist = Twist()

        twist.linear.x = v
        twist.angular.z = w

        self.twist_pub.publish(twist)

    def get_sampled_trajectories(self):
        return self._navigator.get_sampled_trajectories()


    def publish_trajectories(self):
        trajectories = self.get_sampled_trajectories()
        marker_array = MarkerArray()
        for k in range(trajectories.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "sampled_trajectories"
            marker.id = k
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01  # Width of the line
            marker.color.a = 0.4
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            for t in range(trajectories.shape[1]):
                p = trajectories[k, t]
                point = Pose()
                point.position.x = p[0].item()
                point.position.y = p[1].item()
                point.position.z = 0.0
                marker.points.append(point.position)
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


if __name__ == '__main__':
    rospy.init_node('navigation_node')
    rate = rospy.Rate(10)
    navigator_node = NavigationNode()
    rospy.sleep(2)

    visualization = True

    while not rospy.is_shutdown():
        if navigator_node.navigator_ready:
            navigator_node.publish_command()
            if visualization:
                navigator_node.publish_trajectories()
        rate.sleep()
