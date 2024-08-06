import json
import numpy as np
import os
import rospy
import tf
import cv_bridge

from lang_reachability import simulator as sim
from lang_reachability import navigator
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from tqdm import tqdm
from typing import Dict

class SceneReconstruction():
    def __init__(self, result_root):
        self.result_root = result_root
        self.time_line = None
        self.exp_config = self.make_exp_configs()
        self.safe_controller = True

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
        self.combined_map_pub = rospy.Publisher('combined_map_eval', OccupancyGrid, queue_size=1)
        self.semantic_map_pub = rospy.Publisher('semantic_map_eval', OccupancyGrid, queue_size=1)
        self.robot_view_rgb_pub = rospy.Publisher("rgb_eval/image", Image, queue_size=10)

        self.bridge = cv_bridge.CvBridge()
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
        self.semantic_maps_dict = self.load_map_data("semantic_map")
        self.combined_maps_dict = self.load_map_data("combined_map")

        # compute metrics
        self.total_exe_time = self.compute_total_exe_time()
        self.trajectory_cost = self.compute_trajectory_cost()
        self.safety_override_rate = self.compute_safety_override_rate()
        self.average_nominal_planning_time = self.compute_average_nominal_planning_time()
        self.average_safe_planning_time = self.compute_average_safe_planning_time()
        self.average_brt_computation_time = self.compute_average_brt_computation_time()

        if self.safe_controller:
            self.robot_start_time = self.brt_computation_time_raw[0, 1]
        else:
            self.robot_start_time = 0


        self.sim = sim.Simulator(dataset_name=self.dataset_name, test_scene=self.scene_path, test_scene_name=self.scene_name,
                                 initial_state=[self.init_x, self.init_y, self.init_theta], dt=self.exp_config['dt'])

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

    def load_map_data(self, map_name):
        map_dict = {}
        maps = None
        info = None
        if map_name == "semantic_map":
            maps = np.load(os.path.join(self.result_root, "semantic_map_over_time.npy")) # H x W x N
            info = np.load(os.path.join(self.result_root, "semantic_map_times.npy")) # N
        elif map_name == "combined_map":
            maps = np.load(os.path.join(self.result_root, "combined_map_over_time.npy")) # H x W x N
            info = np.load(os.path.join(self.result_root, "combined_map_times.npy")) # N
        for i, (time, resolution, (origin_x, origin_y)) in enumerate(info):
            map_height, map_width= maps[:, :, i].shape
            map_dict[time] = {"data": maps[:, :, i],
                              "resolution": resolution,
                              "origin": (origin_x, origin_y),
                              "shape": (map_height, map_width)}

        return map_dict

    def get_map_at_time(self, map_name, time) -> Dict:
        map_dict = None
        if map_name == "semantic_map":
            map_dict = self.semantic_maps_dict
        elif map_name == "combined_map":
            map_dict = self.combined_maps_dict

        recent_record_time = -1
        for record_time in map_dict.keys():
            if record_time > time:
                break
            else:
                recent_record_time = record_time

        if recent_record_time == -1:
            print("no map is found before the time instance, use the inital map instead")
            first_map = next(iter(map_dict.values()))
            return first_map
        else:
            return map_dict[recent_record_time]


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

    def compute_time_for_action_with_safety(self):
        pass

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
            for [x, y, theta, time_step] in tqdm(self.trajectory_data_raw, desc="Reproducing Trajectory"):
                # get pose
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

                # add pose as a point for path
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0
                self.path_points.points.append(point)

                # get map
                map_data_dict = self.get_map_at_time("combined_map", time_step)
                map_msg = self.construct_map_msg(map_data_dict)

                # get rgb image
                rgb_image = self.sim.get_rgb_from_odom([x, y, 0.0], [0.0, 0.0, theta])
                header = Header()
                header.stamp = rospy.Time.now()
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgba8", header=header)

                # publish
                self.pose_pub.publish(pose_msg)
                self.path_pub.publish(self.path_points)
                self.combined_map_pub.publish(map_msg)
                self.robot_view_rgb_pub.publish(rgb_msg)
                rate.sleep()
        except rospy.ROSInterruptException:
            print("Reproduction interrupted by user.")
            rospy.signal_shutdown("User interrupted the process.")
        except Exception as e:
            print(f"An error occurred: {e}")
            rospy.signal_shutdown("User interrupted the process.")

    def construct_map_msg(self, map_data_dick: Dict):
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = "map"
        map_msg.header.stamp = rospy.Time.now()
        map_msg.data = map_data_dick["data"].flatten()
        map_msg.info.origin.position.x = map_data_dick["origin"][0]
        map_msg.info.origin.position.y = map_data_dick["origin"][1]
        map_msg.info.height = map_data_dick["shape"][0]
        map_msg.info.width = map_data_dick["shape"][1]
        map_msg.info.resolution = map_data_dick["resolution"]

        return map_msg

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
    result_root = "/home/zli133/shared/ml_projects/lang_reachability_ros/results/rtabmap_mppi_vlm_reachability/2024-08-06-17:13:52"

    rospy.init_node("evaluation_node")

    stage = SceneReconstruction(result_root)
    stage.reproduce_trajectory()
    # stage.reproduce_trme("semantic_map", 15)
    #     # stage.reproduce_tajectory()
    stage.print_summary()