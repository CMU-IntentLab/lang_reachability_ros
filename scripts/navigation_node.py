import rospy
import torch
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from lang_reachability import navigator
import argparse
import json

# dir_path = str(Path(__file__).parent.parent)
# logs_dir = os.path.join(dir_path, 'logs')
# if not os.path.exists(logs_dir):
#     os.makedirs(logs_dir)
# logging.basicConfig(filename=os.path.join(logs_dir, 'navigation_node.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NavigationNode:
    def __init__(self, args):
        self.args = args
        self.exp_config = self.make_exp_config()
        self.topics_names = self.make_topics_names()

        self.tf_map_frame = self.exp_config["tf_map_frame"]

        self.path_msg = Path()

        self.path_publisher = rospy.Publisher('/robot_path', Path, queue_size=1)
        self.twist_pub = rospy.Publisher(self.topics_names["nominal_action"], Twist, queue_size=1)
        self.marker_pub = rospy.Publisher(self.topics_names["trajectory_visualization"], MarkerArray, queue_size=1)
        self.planning_time_pub = rospy.Publisher(self.topics_names["nominal_planning_time"], Float32, queue_size=10)

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._dtype = torch.float32
        self._navigator = navigator.Navigator(config=self.exp_config, planner_type="mppi", device=self._device, dtype=self._dtype, dt=0.1)
        self.navigator_ready = False

        self._odom = None
        self._goal = None
        self._map = None

        self._goal_thresh = 0.1

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
    
    def initialize_subs(self):
        # self.odom_sub = rospy.Subscriber(self.topics_names["odom"], Odometry, callback=self.odom_callback)
        self.pose_sub = rospy.Subscriber(self.topics_names["pose"], PoseWithCovarianceStamped, callback=self.pose_callback)
        self.goal_sub = rospy.Subscriber(self.topics_names["goal"], PoseStamped, callback=self.goal_callback)
        self.map_sub = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, callback=self.map_callback)

    def _check_navigator_ready(self):
        if self._odom is not None and self._goal is not None and self._map is not None:
            self.navigator_ready = True
            print(f'begin navigation: goal: ({self._goal.pose.position.x}, {self._goal.pose.position.y})')
        else:
            print(f'data status for navigation: odom: {self._odom is not None}, goal: {self._goal is not None}, map: {self._map is not None}')

    # def odom_callback(self, msg: Odometry) -> None:
    #     self._odom = msg
    #     pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
    #     quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    #     self._navigator.set_odom(position=pos, orientation=quat)
    #     if not self.navigator_ready:
    #         self._check_navigator_ready()
    
    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self._odom = Odometry()
        self._odom.pose = msg.pose
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self._navigator.set_odom(position=pos, orientation=quat)
        if not self.navigator_ready:
            self._check_navigator_ready()  

    def map_callback(self, msg: OccupancyGrid) -> None:
        print('map callback')
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
        start_time = rospy.Time.now().secs
        v, w = self._navigator.get_command()

        time_taken = rospy.Time.now().secs - start_time
        self.planning_time_pub.publish(Float32(data=time_taken))
        # print(v, w)

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
            marker.header.frame_id = self.tf_map_frame
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

    def publish_path(self):
        self.path_msg.header.frame_id = 'map'
        self.path_msg.header.stamp = rospy.Time.now()
        pose_msg = PoseStamped()
        pose_msg.pose = self._odom.pose.pose
        self.path_msg.poses.append(pose_msg)
        self.path_publisher.publish(self.path_msg)

if __name__ == '__main__':
    rospy.init_node('navigation_node')
    rate = rospy.Rate(10)

    parser = argparse.ArgumentParser(description="Command Node")
    parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
    parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
    args = parser.parse_args()

    assert args.topics_path is not None, "topics names json file must be provided"

    navigator_node = NavigationNode(args)
    rospy.sleep(2)
    # wait for the navigator to be initialized before entering callbacks
    navigator_node.initialize_subs()

    visualization = True

    while not rospy.is_shutdown():
        if navigator_node.navigator_ready:
            navigator_node.publish_command()
            if visualization:
                navigator_node.publish_trajectories()
                navigator_node.publish_path()
        rate.sleep()
