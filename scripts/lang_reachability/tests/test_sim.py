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

def display_observation(observation, n=None):
    if n == None:
        n = 1

    img = observation["color_sensor"]
    depth = observation["depth_sensor"]
    # semantic = sample["semantic"]

    arr = [img, depth] #, semantic]
    titles = ["rgba", "depth"] #, "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 1, 1+i)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.savefig(os.path.join(output_root, f'habitat-test-{n}.jpg'))

def run_with_fixed_inputs():
    frames = [] # for storing the generated images
    fig, (ax_rgb, ax_depth, ax_td) = plt.subplots(3, 1)
    # ax_rgb = plt.subplot(1, 2, 1)
    # ax_depth = plt.subplot(1, 2, 2)
    ax_rgb.axis("off")
    ax_depth.axis("off")
    ax_td.axis("off")

    for _ in range(200):
        state = robot.state
        
        img = sim.get_rgb_observation(state[0], state[1], state[2])
        depth = sim.get_depth_observation(state[0], state[1], state[2])
        text = ax_rgb.text(0.05, 0.95, f"x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}",
                   transform=ax_rgb.transAxes, fontsize=12, color='white', backgroundcolor='black')
        
        # topdown_view = sim.get_topdown_view()
        # ax_td.scatter(state[0], state[1], color='red')    # TODO: convert to image coordinates

        frames.append([ax_rgb.imshow(img, animated=True), text, ax_depth.imshow(depth, animated=True)])#, ax_td.imshow(topdown_view, animated=True)])
        # rgb_frames.append([ax_rgb.imshow(img, animated=True)])
        # depth_frames.append([ax_depth.imshow(depth, animated=True)])
        # display_observation(img, i)
        robot.step(0.3, 0.0)
        robot.step(0.0, 0.1)

    print(f'saving animations at {output_root}')
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_root, 'test_sim.mp4'))

def step(v, w):
    robot.step(v, w)
    state = robot.state
    img = sim.get_rgb_observation(state[0], state[1], state[2])
    depth = sim.get_depth_observation(state[0], state[1], state[2])
    # topdown_view = sim.get_topdown_view()
    return state, img, depth

def update_sim(v, w):
    state, img, depth = step(v, w)
    # print(state)
    text = ax_rgb.text(0.05, 0.95, f"x: {state[0]:.2f}, y: {state[1]:.2f}, theta: {state[2]:.2f}",
                   transform=ax_rgb.transAxes, fontsize=12, color='white', backgroundcolor='black')
    img_artist = ax_rgb.imshow(img, animated=True)
    depth_artist = ax_depth.imshow(depth, animated=True)
    # depth_artist = ax_depth.imshow(depth, animated=True)
    frames_rgb.append([img_artist, text])#, depth_artist])
    frames_depth.append([depth_artist])
    # plt.draw()
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

dt = 0.1
sim = simulator.Simulator(dataset_name, test_scene)
x = -0.2
y = -0
theta = 0.0
robot = dubins3d.Dubins3D(init_x=x, init_y=y, init_theta=0, dt=dt)

output_root = os.path.join(dir_path, 'tests/outputs', test_scene_name)
if not os.path.exists(output_root):
    os.mkdir(output_root)

if __name__ == '__main__':
    sim.save_top_down_map(os.path.join(output_root, 'top_down_map.png'))

    # initialize plot stuff
    frames_rgb = [] # for storing the generated images
    frames_depth = []
    # fig, (ax_rgb, ax_depth, ax_td) = plt.subplots(3, 1)
    fig, ax_rgb = plt.subplots(1, 1)
    fig2, ax_depth = plt.subplots(1, 1)
    ax_rgb.axis("off")
    ax_depth.axis("off")

    fig.canvas.mpl_connect('key_press_event', on_press)
    mpl.rcParams['keymap.save'].remove('s')
    mpl.rcParams['keymap.quit'].remove('q')
    plt.show()

    print(f'saving animations at {output_root}')
    ani = animation.ArtistAnimation(fig, frames_rgb, interval=dt*1000, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_root, f'{test_scene_name}_rgb.mp4'))

    ani = animation.ArtistAnimation(fig2, frames_depth, interval=dt*1000, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_root, f'{test_scene_name}_depth.mp4'))
