import sys, os
import json
from pathlib import Path
import numpy as np

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH
import habitat_sim
from simulator import simulator
from systems import dubins3d
from habitat.utils.visualizations import maps
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import matplotlib.animation as animation
from PIL import Image
import random
import magnum as mn
import habitat_sim
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
# from pynput import keyboard

# setup paths for different user
# script_dir = os.path.dirname(os.path.realpath(__file__))

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show()

with open(os.path.join(dir_path, 'configs/path_config.json')) as path_config_file:
    path_config = json.load(path_config_file)

data_root = path_config['data_root']

dataset_name = 'hssd-hab'
scene_idx = 2

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

print(f'test_scene_name: {test_scene_name}')
sim = simulator.Simulator(dataset_name, test_scene).sim
# print("NavMesh area = " + str(sim.pathfinder.navigable_area))
fig3, ax_topdown = plt.subplots(1, 1)
ax_topdown.axis("off")


navmesh_settings = habitat_sim.NavMeshSettings()

# @markdown Choose Habitat-sim defaults (e.g. for point-nav tasks), or custom settings.
use_custom_settings = True  # @param {type:"boolean"}
sim.navmesh_visualization = True  # @param {type:"boolean"}
navmesh_settings.set_defaults()
if use_custom_settings:
    # fmt: off
    #@markdown ---
    #@markdown ## Configure custom settings (if use_custom_settings):
    #@markdown Configure the following NavMeshSettings for customized NavMesh recomputation.
    #@markdown **Voxelization parameters**:
    navmesh_settings.cell_size = 0.05 #@param {type:"slider", min:0.01, max:0.2, step:0.01}
    #default = 0.05
    navmesh_settings.cell_height = 0.2 #@param {type:"slider", min:0.01, max:0.4, step:0.01}
    #default = 0.2

    #@markdown **Agent parameters**:
    navmesh_settings.agent_height = 1.5 #@param {type:"slider", min:0.01, max:3.0, step:0.01}
    #default = 1.5
    navmesh_settings.agent_radius = 0.01 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
    #default = 0.1
    navmesh_settings.agent_max_climb = 0.2 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
    #default = 0.2
    navmesh_settings.agent_max_slope = 85 #@param {type:"slider", min:0, max:85, step:1.0}
    # default = 45.0
    # fmt: on
    # @markdown **Navigable area filtering options**:
    navmesh_settings.filter_low_hanging_obstacles = True  # @param {type:"boolean"}
    # default = True
    navmesh_settings.filter_ledge_spans = True  # @param {type:"boolean"}
    # default = True
    navmesh_settings.filter_walkable_low_height_spans = True  # @param {type:"boolean"}
    # default = True

    # fmt: off
    #@markdown **Detail mesh generation parameters**:
    #@markdown For more details on the effects
    navmesh_settings.region_min_size = 5 #@param {type:"slider", min:0, max:50, step:1}
    #default = 20
    navmesh_settings.region_merge_size = 20 #@param {type:"slider", min:0, max:50, step:1}
    #default = 20
    navmesh_settings.edge_max_len = 12.0 #@param {type:"slider", min:0, max:50, step:1}
    #default = 12.0
    navmesh_settings.edge_max_error = 1.3 #@param {type:"slider", min:0, max:5, step:0.1}
    #default = 1.3
    navmesh_settings.verts_per_poly = 6.0 #@param {type:"slider", min:3, max:6, step:1}
    #default = 6.0
    navmesh_settings.detail_sample_dist = 6.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
    #default = 6.0
    navmesh_settings.detail_sample_max_error = 1.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
    # default = 1.0
    # fmt: on

    # @markdown **Include STATIC Objects**:
    # @markdown Optionally include all instanced RigidObjects with STATIC MotionType as NavMesh constraints.
    navmesh_settings.include_static_objects = True  # @param {type:"boolean"}
    # default = False

navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
if sim.pathfinder.is_loaded:
    navmesh_save_path = os.path.join(output_root, f"{test_scene_name}.navmesh") #@param {type:"string"}
    sim.pathfinder.save_nav_mesh(navmesh_save_path)
    print('Saved NavMesh to "' + navmesh_save_path + '"')
    sim.pathfinder.load_nav_mesh(navmesh_save_path)

height = 0

top_down_map = maps.get_topdown_map(
    sim.pathfinder, map_resolution=1024, height=height
)
recolor_map = np.array(
    [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
)
text = ax_topdown.text(0.05, 0.95, f"height: {height}",
                       transform=ax_topdown.transAxes, fontsize=12, color='white', backgroundcolor='black')
top_down_map = recolor_map[top_down_map]
topdown_artist = ax_topdown.imshow(top_down_map, animated=True)
plt.imshow(top_down_map)
plt.title("top_down_map.png")
plt.savefig(os.path.join(output_root, "top_down_map.png"))
plt.show()


# print("NavMesh area = " + str(sim.pathfinder.navigable_area))
#
# if not navmesh_success:
#     print("Failed to build the navmesh! Try different parameters?")
# else:
#     # @markdown ---
#     # @markdown **Agent parameters**:
#
#     agent_state = sim.agents[0].get_state()
#     set_random_valid_state = False  # @param {type:"boolean"}
#     seed = 5  # @param {type:"integer"}
#     sim.seed(seed)
#     orientation = 0
#     if set_random_valid_state:
#         agent_state.position = sim.pathfinder.get_random_navigable_point()
#         orientation = random.random() * math.pi * 2.0
#     # @markdown Optionally configure the agent state (overrides random state):
#     set_agent_state = True  # @param {type:"boolean"}
#     try_to_make_valid = True  # @param {type:"boolean"}
#     if set_agent_state:
#         pos_x = 0  # @param {type:"number"}
#         pos_y = 0  # @param {type:"number"}
#         pos_z = 0.0  # @param {type:"number"}
#         # @markdown Y axis rotation (radians):
#         orientation = 1.56  # @param {type:"number"}
#         agent_state.position = np.array([pos_x, pos_y, pos_z])
#         if try_to_make_valid:
#             snapped_point = np.array(sim.pathfinder.snap_point(agent_state.position))
#             if not np.isnan(np.sum(snapped_point)):
#                 print("Successfully snapped point to: " + str(snapped_point))
#                 agent_state.position = snapped_point
#     if set_agent_state or set_random_valid_state:
#         agent_state.rotation = utils.quat_from_magnum(
#             mn.Quaternion.rotation(-mn.Rad(orientation), mn.Vector3(0, 1.0, 0))
#         )
#         sim.agents[0].set_state(agent_state)
#
#     agent_state = sim.agents[0].get_state()
#     print("Agent state: " + str(agent_state))
#     print(" position = " + str(agent_state.position))
#     print(" rotation = " + str(agent_state.rotation))
#     print(" orientation (about Y) = " + str(orientation))
#
#     observations = sim.get_sensor_observations()
#     rgb = observations["color_sensor"]
#     # semantic = observations["semantic_sensor"]
#     depth = observations["depth_sensor"]
#
#     display = True
#     if display:
#         # display_sample(rgb, semantic, depth)
#         # @markdown **Map parameters**:
#         # fmt: off
#         meters_per_pixel = 0.025  # @param {type:"slider", min:0.01, max:0.1, step:0.005}
#         # fmt: on
#         agent_pos = agent_state.position
#         # topdown map at agent position
#         top_down_map = maps.get_topdown_map(
#             sim.pathfinder, height=agent_pos[1], meters_per_pixel=meters_per_pixel
#         )
#         recolor_map = np.array(
#             [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
#         )
#         top_down_map = recolor_map[top_down_map]
#         grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
#         # convert world agent position to maps module grid point
#         agent_grid_pos = maps.to_grid(
#             agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=sim.pathfinder
#         )
#         agent_forward = utils.quat_to_magnum(
#             sim.agents[0].get_state().rotation
#         ).transform_vector(mn.Vector3(0, 0, -1.0))
#         agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
#         # draw the agent and trajectory on the map
#         maps.draw_agent(
#             top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8
#         )
#         print("\nDisplay topdown map with agent:")
#         display_map(top_down_map)