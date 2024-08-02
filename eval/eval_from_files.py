import json
import numpy as np
import os

from lang_reachability import simulator as sim


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
        self.sim = sim.Simulator(dataset_name=self.dataset_name, test_scene=self.scene_path, test_scene_name=self.scene_name,
                                 initial_state=[self.init_x, self.init_y, self.init_theta], dt=self.exp_config['dt'])

        self.trajectory_data = np.load(os.path.join(result_root, "trajectory.npy"))
        print(self.trajectory_data)

    def initiate_time_line(self):
        pass

    def make_exp_configs(self):
        exp_config_path = os.path.join(self.result_root, "exp_config.json")
        with open(exp_config_path, "r") as f:
            exp_config = json.load(f)
        return exp_config

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


if __name__ == '__main__':
    result_root = "/home/zli133/shared/ml_projects/lang_reachability_ros/results/rtabmap_mppi_vlm_reachability/2024-08-02-15:04:37"
    stage = SceneReconstruction(result_root)

