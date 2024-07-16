import sys, os
import json
from pathlib import Path

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH

from simulator import simulator
from systems import dubins3d

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


# setup paths for different user
# script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'configs/path_config.json')) as path_config_file:
    path_config = json.load(path_config_file)

data_root = path_config['data_root']
output_root = os.path.join(dir_path, 'tests/outputs')
if not os.path.exists(output_root):
    os.mkdir(output_root)


robot = dubins3d.Dubins3D()

frames = [] # for storing the generated images
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

for i in range(500):
    state = robot.state
    frames.append([ax.scatter(state[0], state[1], color='blue', animated=True)])
    robot.step(0.5, 0.5)  # TODO: why do we have multiply by -1 to work?


ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
ani.save(os.path.join(output_root, 'test_dyanmics.mp4'))