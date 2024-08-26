import rospy
import json
import argparse

from nav_msgs.msg import OccupancyGrid

class VisualizationNode:
    def __init__(self, args):
        self.args = args
        self.exp_configs = self.make_exp_config()
        self.topics_names = self.make_topics_names()
        self.tf_map_frame = self.exp_configs["tf_map_frame"]
        # define subscribers
        self.vlm_map_sub = rospy.Subscriber(self.topics_names["semantic_grid_map"], OccupancyGrid, self.vlm_map_callback)
        self.brt_map = rospy.Subscriber(self.topics_names["viz_brt"], OccupancyGrid, self.brt_map_callback)
        # self.floorplan = rospy.Subscriber(self.topics_names["grid_map"], OccupancyGrid, self.floorplan_callback)
        # define publishers
        self.vlm_map_viz_pub = rospy.Publisher(self.topics_names["viz_vlm_map"], OccupancyGrid, queue_size=1)
        self.brt_map_viz_pub = rospy.Publisher(self.topics_names["viz_brt_colored"], OccupancyGrid, queue_size=1)

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

    def vlm_map_callback(self, msg: OccupancyGrid):
        # recolor the map
        data = list(msg.data)
        data = [1 if value == 100 else value for value in data]
        msg.data = tuple(data)
        self.vlm_map_viz_pub.publish(msg)

    def brt_map_callback(self, msg: OccupancyGrid):
        data = list(msg.data)
        data = [90 if value == 100 else value for value in data]
        msg.data = tuple(data)
        self.brt_map_viz_pub.publish(msg)

if __name__ == '__main__':
        rospy.init_node('visualization_node')

        parser = argparse.ArgumentParser(description="Visualization Node")
        parser.add_argument('--exp_path', type=str, default=None, help='path to experiment json file')
        parser.add_argument('--topics_path', type=str, default=None, help='path to ROS topics names json file')
        args = parser.parse_args()

        assert args.topics_path is not None, "topics names json file must be provided"

        navigator_node = VisualizationNode(args)
        rospy.sleep(2)

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()