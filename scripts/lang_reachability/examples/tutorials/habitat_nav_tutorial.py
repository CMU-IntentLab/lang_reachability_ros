import math
import os
import random

import git
import imageio
import magnum as mn
import json
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# repo = git.Repo(".", search_parent_directories=True)
# setup paths for different user
with open('path_config.json') as path_config_file:
    path_config = json.load(path_config_file)

data_root = path_config['data_root']
dir_path = path_config['dir_path']

print(f"data_path = {data_root}")
# @markdown Optionally configure the save path for video output:
output_directory = os.path.join(
    dir_path, "examples/tutorials/nav_output/"
)  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)


# This is the scene we are going to load.
# we support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
test_scene = os.path.join(
    data_root, "scene_datasets/versioned_data/habitat_test_scenes/apartment_1.glb"
)

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0.8,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}

# Only supports RGBA
def display_sample(sample, n=None):
    if n == None:
        n = 1

    img = sample
    # img = sample["rgba"]
    # depth = sample["depth"]
    # semantic = sample["semantic"]

    arr = [img] #, depth] #, semantic]
    titles = ["rgba"] #, "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.savefig(f'/home/leo/git/lang-reachability/habitat-test-{n}.jpg')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    do_make_video = args.make_video
else:
    show_video = False
    do_make_video = False
    display = False

# NOTE: commented out because cannot resolve import
# # import the maps module alone for topdown mapping
# if display:
#     from habitat.utils.visualizations import maps


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)

display = True
def navigateAndSee(action="", n=None):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"], n)


n = 0
action = "turn_right"
navigateAndSee(action, n)
n += 1

action = "turn_right"
navigateAndSee(action, n)
n += 1

action = "move_forward"
navigateAndSee(action, n)
n += 1

action = "turn_left"
navigateAndSee(action, n)

# action = "move_backward"   // #illegal, no such action in the default action space
# navigateAndSee(action)