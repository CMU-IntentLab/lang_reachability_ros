import os
import numpy as np
from matplotlib import pyplot as plt

data_dir = "/home/leo/riss_ws/src/lang_reachability_ros/experiments/hardware/scenario1"

traj = np.load(os.path.join(data_dir, "trajectory.npy"))
print(traj.shape)

plt.plot(traj[:, 0], traj[:, 1])
plt.show()

# print(brt_computation_time)

# def load_data(data_dir):
    # pass