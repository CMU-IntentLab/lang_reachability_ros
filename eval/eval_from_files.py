import json
import numpy as np
import os

from lang_reachability import simulator as sim
from lang_reachability import navigator


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

        # get data ready
        self.trajectory_data_raw = None
        self.value_function_data_raw = None
        self.trajectory_dict = self.load_trajectory_data()
        self.value_function_dict= self.load_value_function_data()

        # compute metrics
        self.total_exe_time = self.compute_total_exe_time()

        # self.sim = sim.Simulator(dataset_name=self.dataset_name, test_scene=self.scene_path, test_scene_name=self.scene_name,
        #                          initial_state=[self.init_x, self.init_y, self.init_theta], dt=self.exp_config['dt'])

    def initiate_time_line(self):
        pass

    def load_trajectory_data(self):
        self.trajectory_data_raw = np.load(os.path.join(result_root, "trajectory.npy"))
        trajectory_dict= {int(row[3]): row[:3].tolist() for row in self.trajectory_data_raw}
        return trajectory_dict

    def load_value_function_data(self):
        self.value_function_data_raw = np.load(os.path.join(result_root, "value_function_at_state.npy"))
        value_function_dict = {int(row[-1]): row[-2] for row in self.value_function_data_raw}
        return value_function_dict

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
        print(total_cost)
        return total_cost

    def compute_total_exe_time(self):
        return self.trajectory_data_raw[-1, -1] - self.trajectory_data_raw[0, -1]




if __name__ == '__main__':
    result_root = "/home/zli133/shared/ml_projects/lang_reachability_ros/results/rtabmap_mppi_vlm_reachability/2024-08-04-11:27:26"
    stage = SceneReconstruction(result_root)