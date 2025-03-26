import rospy
import tf2_ros
import os
import json
import logging
import tf.transformations
import sys
import argparse

from pathlib import Path
from geometry_msgs.msg import PoseStamped, Quaternion


class CommandNode:
    def __init__(self, args):
        self.args = args
        self.exp_config = self.make_exp_config()
        self.topics_names = self.make_topics_names()
        self.goal_pub = rospy.Publisher(self.topics_names["goal"], PoseStamped, queue_size=1)
        self.goal = self.exp_config['goal']
        print(f"command node initialized with goal = x={self.goal[0]}, y={self.goal[1]}")

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

    def send_goal(self):
        theta = 0.0
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = self.goal[0]
        goal.pose.position.y = self.goal[1]
        goal.pose.position.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        goal.pose.orientation = Quaternion(*quat)
        # self.logger.info(f"send goal: x={x}, y={y}, theta={theta}")
        # print(f"sending goal: x={self.goal[0]}, y={self.goal[1]}, theta={theta}")
        self.goal_pub.publish(goal)
        rospy.sleep(1)
        # self.logger.info('goal published')

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

        return logger


parser = argparse.ArgumentParser(description="Command Node")
parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
args = parser.parse_args()

assert args.exp_path is not None, "a experiment config file must be provided"
assert args.topics_path is not None, "topics names file must be provided"

rospy.init_node('command')
rate = rospy.Rate(10)
cmd_node = CommandNode(args)
# cmd_node = CommandNode(None)

while not rospy.is_shutdown():
    cmd_node.send_goal()
    rate.sleep()
