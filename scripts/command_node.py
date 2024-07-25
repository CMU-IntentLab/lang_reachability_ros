import rospy
import tf2_ros
import tf.transformations

from geometry_msgs.msg import PoseStamped, Quaternion

class CommandNode:
    def __init__(self):
        self.language_cmd = "avoid the rug" # TODO: handle for owl-vit

        self.goal_pub = rospy.Publisher("goal", PoseStamped, queue_size=1)
    def send_goal(self):
        # x = 1.0
        # y = -3.5
        x = 3.0
        y = 2.5
        theta = 0.0
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "odom"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        goal.pose.orientation = Quaternion(*quat)
        print(f"send goal: x={x}, y={y}, theta={theta}")
        self.goal_pub.publish(goal)
        rospy.sleep(1)
        print('goal published')


if __name__ == '__main__':
    rospy.init_node('command')
    rate = rospy.Rate(10)
    cmd_node = CommandNode()

    while not rospy.is_shutdown():
        cmd_node.send_goal()
        rate.sleep()