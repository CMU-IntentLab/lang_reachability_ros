
import habitat
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH

from typing import cast
from habitat.utils.visualizations import maps
import matplotlib.animation as animation
from simulator import simulator

if __name__ == '__main__':
    with open(os.path.join(dir_path, 'configs/path_config.json')) as path_config_file:
        path_config = json.load(path_config_file)

    data_root = path_config['data_root']
    dataset_name = 'hssd-hab'
    scene_idx = 0
    if dataset_name == 'hssd':
        scene_map = {0: "102344469", 1: "102344022", 2: "102344094", 3: "103997403_171030405", 4: "102815859",
                     5: "102816216"}
        test_scene_name = scene_map[scene_idx]
        test_scene = os.path.join(data_root, "hssd", f"{test_scene_name}.glb")
    elif dataset_name == 'hssd-hab':
        scene_map = {0: "102344469", 1: "102344022", 2: "102344094", 3: "103997403_171030405", 4: "102815859",
                     5: "102816216", 6: "102344094_raw", 7: "102344094_mod"}
        test_scene_name = scene_map[scene_idx]
        test_scene = test_scene_name
    elif dataset_name == 'hm3d':
        scene_map = {0: "00099-226REUyJh2K", 1: "00013-sfbj7jspYWj", 2: "00198-eZrc5fLTCmi"}
        test_scene_name = scene_map[scene_idx]
        sub_title = test_scene_name.split('-')[-1]
        test_scene = os.path.join(data_root, "hm3d/train", f"{test_scene_name}/{sub_title}.basis.glb")

    output_root = os.path.join(dir_path, 'tests/outputs', test_scene_name)
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    sim = simulator.Simulator(dataset_name, test_scene)
    topdown_save_path = os.path.join(output_root, 'top_down_map.png')
    sim.save_top_down_map(topdown_save_path)
    print(f'topdown map saved to {topdown_save_path}')