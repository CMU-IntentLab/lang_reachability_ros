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
    update_occ_map(state, img, depth)
    # test_ext_mat(state)
    # test_img_projection(state, img)

def test_ext_mat(state):
    T = sim.get_camera_extrinsics_mat(state)
    pw = np.array([state[0], state[1], 1, 1])
    pc = np.matmul(T, pw)
    print(pc)

def test_img_projection(state, img):
    u, v = sim.world_to_pixel(x=0.31, y=5.02, z=0.9, robot_state=state)
    # text = ax_rgb.text(0.05, 0.95, f"x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}",
    #                transform=ax_rgb.transAxes, fontsize=12, color='white', backgroundcolor='black')
    img_artist = ax_rgb.imshow(img, animated=True)
    proj = ax_rgb.scatter(u, v, color="red")
    # frames_rgb.append([img_artist, text, proj])
    frames_rgb.append([img_artist, proj])
    fig.canvas.flush_events()
    plt.pause(0.01)

def position_to_pixel(x_pos, y_pos, origin='upper'):
    x_pix = x_pos/map_res + map_size/2
    y_pix = y_pos/map_res
    if origin == 'upper':
        y_pix = map_size - y_pix
    return x_pix.astype(int), y_pix.astype(int)

def update_occ_map(state, img, depth):
    detections = det.detect(img)
    for bbox, label in detections:
        x_occ, y_occ = sim.estimate_object_position(state, depth, bbox)
        x_occ_idx, y_occ_idx = position_to_pixel(x_occ, y_occ)
        occ_map[y_occ_idx, x_occ_idx] = 255
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        depth_artist = ax_depth.imshow(depth[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        frames_depth.append([depth_artist])

    text = ax_rgb.text(0.05, 0.95, f"x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}",
                   transform=ax_rgb.transAxes, fontsize=12, color='white', backgroundcolor='black')
    img_artist = ax_rgb.imshow(img, animated=True)
    # depth_artist = ax_depth.imshow(depth, animated=True)
    map_artist = ax_map.imshow(occ_map, animated=True, cmap='gray', origin='upper')
    position = ax_map.scatter(state[0]/map_res + map_size/2, map_size - state[1]/map_res, color="red", s=5)
    frames_rgb.append([img_artist, text, position, map_artist])
    frames_map.append([map_artist])
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
    scene_idx = 3
    test_scene_name = scene_map[scene_idx]
    test_scene = os.path.join(data_root, "hssd", f"{test_scene_name}.glb")
elif dataset_name == 'hm3d':
    scene_map = {0: "00099-226REUyJh2K"}
    test_scene_name = scene_map[scene_idx]
    sub_title = test_scene_name.split('-')[-1]
    test_scene = os.path.join(data_root, "hm3d/train", f"{test_scene_name}/{sub_title}.basis.glb")


x = 0.31; y = 3.5; theta = 0.0; dt = 0.1; score_threshold = 0.18
robot = dubins3d.Dubins3D(init_x=x, init_y=y, init_theta=theta, dt=dt)
sim = simulator.Simulator(dataset_name, test_scene)
det = object_detector.ObjectDetector(model_name="google/owlv2-base-patch16", score_threshold=score_threshold)
det.add_new_text_query("stay away from kids toys")

output_root = os.path.join(dir_path, 'tests/outputs', test_scene_name)
if not os.path.exists(output_root):
    os.mkdir(output_root)

if __name__ == '__main__': 
    map_res = 0.1   # each pixel = 0.1 meters
    map_size_meters = 15
    map_size = int(map_size_meters/map_res)   # in pixels
    occ_map = np.zeros((map_size, map_size))
    
    # initialize plot stuff
    frames_rgb = [] # for storing the generated images
    frames_depth = []
    frames_map = []
    fig, (ax_rgb, ax_map, ax_depth) = plt.subplots(1, 3)

    # uncomment if you want to save each axes as a separate animation #
    # fig, ax_rgb = plt.subplots(1, 1)
    # fig2, ax_depth = plt.subplots(1, 1)
    # fig3, ax_map = plt.subplots(1, 1)

    fig.canvas.mpl_connect('key_press_event', on_press)
    mpl.rcParams['keymap.save'].remove('s')
    mpl.rcParams['keymap.quit'].remove('q')
    plt.show()

    print(f'saving animations at {output_root}')
    ani = animation.ArtistAnimation(fig, frames_rgb, interval=dt*1000, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_root, f'{test_scene_name}_rgb.mp4'))

    # uncomment if you want to save each axes as a separate animation #
    # ani = animation.ArtistAnimation(fig2, frames_depth, interval=dt*1000, blit=True, repeat_delay=1000)
    # ani.save(os.path.join(output_root, f'{test_scene_name}_depth.mp4'))
    # ani = animation.ArtistAnimation(fig3, frames_map, interval=dt*1000, blit=True, repeat_delay=1000)
    # ani.save(os.path.join(output_root, f'{test_scene_name}_map.mp4'))