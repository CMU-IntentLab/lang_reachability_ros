#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np


class SyntheticCostmapPublisher:
    def __init__(self):
        # Initialize the node
        rospy.init_node('synthetic_costmap_publisher', anonymous=True)

        # Publisher for the synthetic costmap
        self.costmap_pub = rospy.Publisher('/synthetic_costmap', OccupancyGrid, queue_size=10)

        # Set the costmap parameters
        self.resolution = 0.1  # 10 cm per cell
        self.width = 10  # 100 cells wide
        self.height = 10  # 100 cells tall
        self.origin_x = -3.0  # Origin in meters
        self.origin_y = -4.0

        # Define the rate of publishing
        self.rate = rospy.Rate(1)  # 1 Hz

    def create_synthetic_costmap(self):
        # Create a new OccupancyGrid message
        costmap = OccupancyGrid()

        # Set the header
        costmap.header = Header()
        costmap.header.stamp = rospy.Time.now()
        costmap.header.frame_id = "map"

        # Set the map metadata
        costmap.info.resolution = self.resolution
        costmap.info.width = self.width
        costmap.info.height = self.height
        costmap.info.origin.position.x = self.origin_x
        costmap.info.origin.position.y = self.origin_y
        costmap.info.origin.position.z = 0.0
        costmap.info.origin.orientation.w = 1.0  # No rotation

        # Generate synthetic data for the costmap
        # Example: Create a gradient pattern from 0 to 100
        data = np.zeros((self.height, self.width), dtype=np.int8)
        for y in range(self.height):
            for x in range(self.width):
                # Creating a simple gradient pattern
                data[y, x] = int((x + y) / (self.width + self.height - 2) * 100)

        # Flatten the data to a 1D list
        costmap.data = data.flatten().tolist()

        return costmap

    def run(self):
        while not rospy.is_shutdown():
            # Create and publish the synthetic costmap
            synthetic_costmap = self.create_synthetic_costmap()
            self.costmap_pub.publish(synthetic_costmap)

            # Sleep for the defined rate
            self.rate.sleep()


if __name__ == '__main__':
    try:
        node = SyntheticCostmapPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
