import rospy
import tf2_ros
import os
import logging
import tf.transformations
import sys

from pathlib import Path
from geometry_msgs.msg import PoseStamped, Quaternion


class CommandNode:
    def __init__(self):
        self.language_cmd = "avoid the rug" # TODO: handle for owl-vit
        self.goal_pub = rospy.Publisher("goal", PoseStamped, queue_size=1)
        # self.logger = self.setup_logger(log_file)
        # self.logger.info('command node initialized')
        print("command node initialized")

    def send_goal(self):
        x = 3.5
        y = 2
        theta = 0.0
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "odom"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        goal.pose.orientation = Quaternion(*quat)
        # self.logger.info(f"send goal: x={x}, y={y}, theta={theta}")
        print(f"sending goal: x={x}, y={y}, theta={theta}")
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


if __name__ == '__main__':
    print('test')

    rospy.init_node('command')
    rate = rospy.Rate(10)
    cmd_node = CommandNode()

    while not rospy.is_shutdown():
        cmd_node.send_goal()
        rate.sleep()
