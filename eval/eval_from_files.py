import json
import numpy as np
import os
import rospy
import tf

from lang_reachability import simulator as sim
from lang_reachability import navigator
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from tqdm import tqdm

class SceneReconstruction():
    def __init__(self, result_root):
        self.result_root = result_root
        self.time_line = None
        self.exp_config = self.make_exp_configs()

        self.data_root = self.exp_config['data_root']
        self.dataset_name = self.exp_config['dataset_name']
        self.scene_name = self.exp_config['scene_name']
        self.scene_path = self.make_scene_path()
        self.init_x = self.exp_config['initial_pose'][0]
        self.init_y = self.exp_config['initial_pose'][1]
        self.init_theta = self.exp_config['initial_pose'][2]
        self.goal = self.exp_config['goal']

        self.pose_pub = rospy.Publisher('pose_eval', PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher('path_marker_eval', Marker, queue_size=1)

        self.path_points = Marker()
        self.initiate_path_points()

        # get data ready
        self.trajectory_data_raw = None
        self.value_function_data_raw = None
        self.brt_computation_time_raw = None
        self.nominal_planning_time_raw = None
        self.safe_planning_time_raw = None
        self.trajectory_dict, self.total_time_steps = self.load_trajectory_data()
        self.value_function_dict= self.load_value_function_data()
        self.brt_computation_time_dict = self.load_brt_computation_time()
        self.nominal_planning_time_dict = self.load_nominal_planning_time()
        self.safe_planning_time_dict = self.load_safe_planning_time()

        # compute metrics
        self.total_exe_time = self.compute_total_exe_time()
        self.trajectory_cost = self.compute_trajectory_cost()
        self.safety_override_rate = self.compute_safety_override_rate()
        self.average_nominal_planning_time = self.compute_average_nominal_planning_time()
        self.average_safe_planning_time = self.compute_average_safe_planning_time()
        self.average_brt_computation_time = self.compute_average_brt_computation_time()


        # self.sim = sim.Simulator(dataset_name=self.dataset_name, test_scene=self.scene_path, test_scene_name=self.scene_name,
        #                          initial_state=[self.init_x, self.init_y, self.init_theta], dt=self.exp_config['dt'])

    def initiate_time_line(self):
        pass

    def initiate_path_points(self):
        self.path_points.header.frame_id = "map"
        self.path_points.header.stamp = rospy.Time.now()
        self.path_points.ns = "trajectory"
        self.path_points.id = 0
        self.path_points.type = Marker.POINTS
        self.path_points.action = Marker.ADD
        self.path_points.scale.x = 0.1
        self.path_points.scale.y = 0.1
        self.path_points.color.a = 1.0
        self.path_points.color.r = 1.0
        self.path_points.color.g = 0.0
        self.path_points.color.b = 0.0
        self.path_points.points = []

    def load_nominal_planning_time(self):
        self.nominal_planning_time_raw = np.load(os.path.join(self.result_root, "nominal_planning_time.npy"))
        nominal_planning_time_dict = {int(row[-1]): row[0] for row in self.nominal_planning_time_raw}
        return nominal_planning_time_dict

    def load_safe_planning_time(self):
        self.safe_planning_time_raw = np.load(os.path.join(self.result_root, "safe_planning_time.npy"))
        safe_planning_time_dict = {int(row[-1]): row[0] for row in self.safe_planning_time_raw}
        return safe_planning_time_dict

    def load_trajectory_data(self):
        self.trajectory_data_raw = np.load(os.path.join(self.result_root, "trajectory.npy"))
        total_time_steps = self.trajectory_data_raw.shape[0]
        trajectory_dict= {int(row[3]): row[:3].tolist() for row in self.trajectory_data_raw}
        return trajectory_dict, total_time_steps

    def load_value_function_data(self):
        self.value_function_data_raw = np.load(os.path.join(self.result_root, "value_function_at_state.npy"))
        value_function_dict = {int(row[-1]): row[-2] for row in self.value_function_data_raw}
        return value_function_dict

    def load_brt_computation_time(self):
        self.brt_computation_time_raw = np.load(os.path.join(self.result_root, "brt_computation_time.npy"))
        brt_computation_time_dict = {int(row[-1]): row[0] for row in self.brt_computation_time_raw}
        return brt_computation_time_dict

    def load_combined_map_data(self):
        self.combined_map_data_raw = np.load(os.path.join(self.result_root, "combined_map_over_time.npy"))
        print(type(self.combined_map_data_raw))

    def make_exp_configs(self):
        exp_config_path = os.path.join(self.result_root, "exp_config.json")
        with open(exp_config_path, "r") as f:
            exp_config = json.load(f)
        return exp_config

    def compute_safety_override_rate(self):
        global_counter = 0
        override_counter = 0
        for time_step in self.value_function_data_raw:
            value = time_step[-2]
            if value < 0:
                override_counter += 1
            global_counter += 1

        return override_counter / global_counter


    def make_scene_path(self):
        scene_path = None
        if self.dataset_name == 'hssd':
            scene_path = os.path.join(self.data_root, "hssd", f"{self.scene_name}.glb")
        elif self.dataset_name == 'hssd-hab':
            scene_path = self.scene_name
        elif self.dataset_name == 'hm3d':
            sub_title = self.scene_name.split('-')[-1]
            scene_path = os.path.join(self.data_root, "hm3d/train", f"{self.scene_name}/{sub_title}.basis.glb")
        return scene_path

    def compute_average_time(self):
        # TODO: find a better way to average the BRT time
        pass

    def compute_trajectory_cost(self):
        # TODO: also consider time for cost, like taxi
        x_coords = self.trajectory_data_raw[:, 0]
        y_coords = self.trajectory_data_raw[:, 1]

        goal_x = self.goal[0]
        goal_y = self.goal[1]

        x_diff_squared = (x_coords - goal_x) ** 2
        y_diff_squared = (y_coords - goal_y) ** 2

        distances = np.sqrt(x_diff_squared + y_diff_squared)
        total_cost = np.sum(distances)
        return total_cost

    def compute_total_exe_time(self):
        return self.trajectory_data_raw[-1, -1] - self.trajectory_data_raw[0, -1]

    def compute_average_nominal_planning_time(self):
        return np.mean(self.nominal_planning_time_raw[:, 0])

    def compute_average_brt_computation_time(self):
        return np.mean(self.brt_computation_time_raw[:, 0])

    def compute_average_safe_planning_time(self):
        return np.mean(self.safe_planning_time_raw[:, 0])

    def reproduce_trajectory(self):
        rate = rospy.Rate(1)
        try:
            for [x, y, theta, time_step] in tqdm(self.trajectory_data_raw[-40:], desc="Reproducing Trajectory"):
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"

                pose_msg.pose.position.x = x
                pose_msg.pose.position.y = y
                pose_msg.pose.position.z = 0.0

                quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
                pose_msg.pose.orientation.x = quaternion[0]
                pose_msg.pose.orientation.y = quaternion[1]
                pose_msg.pose.orientation.z = quaternion[2]
                pose_msg.pose.orientation.w = quaternion[3]

                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0
                self.path_points.points.append(point)

                self.pose_pub.publish(pose_msg)
                self.path_pub.publish(self.path_points)
                rate.sleep()
        except rospy.ROSInterruptException:
            print("Reproduction interrupted by user.")
            rospy.signal_shutdown("User interrupted the process.")
        except Exception as e:
            print(f"An error occurred: {e}")
            rospy.signal_shutdown("User interrupted the process.")


    def print_summary(self):
        print("**************************************************************")
        print("experiment summery: ")
        print(f"Safety Override Rate:          {self.safety_override_rate:10.2f}")
        print(f"Trajectory Cost:               {self.trajectory_cost:10.2f}")
        print(f"Average Nominal Planning Time: {self.average_nominal_planning_time:10.2f}s")
        print(f"Average Safety Planning Time:  {self.average_safe_planning_time:10.2f}s")
        print(f"Average BRT Computation Time:  {self.average_brt_computation_time:10.2f}s")
        print(f"Total Execution Time:          {self.total_exe_time:10.2f}s")
        print("**************************************************************")


if __name__ == '__main__':
    result_root = "/home/zli133/shared/ml_projects/lang_reachability_ros/results/rtabmap_mppi_vlm_reachability/2024-08-04-11:27:26"

    rospy.init_node("evaluation_node")

    stage = SceneReconstruction(result_root)
    stage.reproduce_trajectory()
    stage.print_summary()