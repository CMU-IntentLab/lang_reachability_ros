import sys, os
import json
from pathlib import Path

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH

from simulator import simulator
from systems import dubins3d
from perception import object_detector

import numpy as np
import cv2
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

def step(v, w):
    robot.step(v, w)
    state = robot.state
    img = sim.get_rgb_observation(state[0], state[1], state[2])
    depth = sim.get_depth_observation(state[0], state[1], state[2])
    # topdown_view = sim.get_topdown_view()
    return state, img, depth

def update_sim(v, w):
    state, img, depth = step(v, w)
    
    detections = det.detect(img)
    for bbox, label in detections:
        x_occ, y_occ = det.estimate_object_position(state, depth, bbox)
        x_occ_idx = x_occ/map_res
        y_occ_idx = y_occ/map_res
        occ_map[x_occ_idx.int(), y_occ_idx.int()] = 255

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    text = ax_rgb.text(0.05, 0.95, f"x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}",
                   transform=ax_rgb.transAxes, fontsize=12, color='white', backgroundcolor='black')
    img_artist = ax_rgb.imshow(img, animated=True)
    # depth_artist = ax_depth.imshow(depth, animated=True)
    map_artist = ax_map.imshow(occ_map, animated=True, cmap='gray')
    position = ax_map.scatter(state[1]/map_res, state[0]/map_res, color="blue")
    frames_rgb.append([img_artist, text])
    # frames_depth.append([depth_artist])
    frames_map.append([map_artist, position])
    fig.canvas.flush_events()
    plt.pause(0.01)

def on_press(event):
    try:
        if event.key == 'w':
            update_sim(0.8, 0.0)
        elif event.key == 's':
            update_sim(-0.8, 0.0)
        elif event.key == 'a':
            update_sim(0.0, 0.5)
        elif event.key == 'd':
            update_sim(0.0, -0.5)
        elif event.key == 'q':
            update_sim(0.8, 0.5)
        elif event.key == 'e':
            update_sim(0.8, -0.5)
    except AttributeError:
        pass

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


x = 0.31; y = 3.5; theta = 1.6; dt = 0.1
robot = dubins3d.Dubins3D(init_x=x, init_y=y, init_theta=theta, dt=dt)
sim = simulator.Simulator(dataset_name, test_scene)
det = object_detector.ObjectDetector(model_name="google/owlv2-base-patch16", score_threshold=0.3)
det.add_new_text_query("stay away from the ladder")

output_root = os.path.join(dir_path, 'tests/outputs', test_scene_name)
if not os.path.exists(output_root):
    os.mkdir(output_root)

if __name__ == '__main__':
    # run_with_fixed_inputs()
    # exit()
    
    map_res = 0.1   # each pixel = 0.1 meters
    occ_map = np.zeros((100, 100))
    
    # initialize plot stuff
    frames_rgb = [] # for storing the generated images
    frames_depth = []
    frames_map = []
    fig, ax_rgb = plt.subplots(1, 1)
    fig2, ax_depth = plt.subplots(1, 1)
    fig3, ax_map = plt.subplots(1, 1)
    ax_rgb.axis("off")
    ax_depth.axis("off")

    fig.canvas.mpl_connect('key_press_event', on_press)
    mpl.rcParams['keymap.save'].remove('s')
    mpl.rcParams['keymap.quit'].remove('q')
    plt.show()

    print(f'saving animations at {output_root}')
    ani = animation.ArtistAnimation(fig, frames_rgb, interval=dt*1000, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_root, f'{test_scene_name}_rgb.mp4'))

    # ani = animation.ArtistAnimation(fig2, frames_depth, interval=dt*1000, blit=True, repeat_delay=1000)
    # ani.save(os.path.join(output_root, f'{test_scene_name}_depth.mp4'))

    ani = animation.ArtistAnimation(fig3, frames_map, interval=dt*1000, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_root, f'{test_scene_name}_map.mp4'))
