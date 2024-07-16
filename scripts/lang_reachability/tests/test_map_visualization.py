import sys, os
import json
from pathlib import Path

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH

from simulator import simulator
from systems import dubins3d

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# from pynput import keyboard

# setup paths for different user
# script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'configs/path_config.json')) as path_config_file:
    path_config = json.load(path_config_file)

data_root = path_config['data_root']


def get_topdown_view():
    topdown_view = sim.get_topdown_view()
    fig3, ax_td = plt.subplots(1, 1)
    ax_td.imshow(topdown_view)
    plt.show()


dataset_name = 'hssd'
scene_idx = 0
if dataset_name == 'hssd':
    scene_map = {0: "102344469", 1: "102344022", 2: "102344094", 3: "103997403_171030405", 4: "102815859", 5:"102816216"}
    scene_idx = 2
    test_scene_name = scene_map[scene_idx]
    test_scene = os.path.join(data_root, "hssd", f"{test_scene_name}.glb")
elif dataset_name == 'hm3d':
    scene_map = {0: "00099-226REUyJh2K"}
    test_scene_name = scene_map[scene_idx]
    sub_title = test_scene_name.split('-')[-1]
    test_scene = os.path.join(data_root, "hm3d/train", f"{test_scene_name}/{sub_title}.basis.glb")

sim = simulator.Simulator(dataset_name, test_scene)

output_root = os.path.join(dir_path, 'tests/outputs', test_scene_name)
if not os.path.exists(output_root):
    os.mkdir(output_root)

if __name__ == '__main__':
    get_topdown_view()
    exit()
