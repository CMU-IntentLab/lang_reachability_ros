import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture
from odp.dynamics import DubinsCar4D2, DubinsCar4D
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

import skfmm
import math
import matplotlib.pyplot as plt

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

def block_average(original_array, block_size):
    new_rows = original_array.shape[0] // block_size * block_size
    new_cols = original_array.shape[1] // block_size * block_size
    original_array = original_array[:new_rows, :new_cols]
    shape = (original_array.shape[0] // block_size, original_array.shape[1] // block_size)
    downsampled_array = original_array.reshape(shape[0], block_size, shape[1], block_size).mean(axis=(1, 3))
    return downsampled_array

def get_initial_settings():
    # STEP 1: Define grid
    grid_map = np.load("/home/leo/git/hj_reachability/top_down_map.npy").astype(float)
    grid_map = block_average(grid_map, 10)
    grid_shape = grid_map.shape
    plt.imshow(grid_map.T, origin='lower')

    # STEP 1: Define grid
    grid_min = np.array([-10.0, -10.0, -math.pi])
    grid_max = np.array([10.0, 10.0, math.pi])
    dims = 3
    N = np.array([grid_shape[0], grid_shape[1], 150])
    pd=[2]
    g = Grid(grid_min, grid_max, dims, N, pd)

    # STEP 2: Generate initial values for grid using occupancy map
    initial_values = grid_map - 0.5
    initial_values = skfmm.distance(initial_values, dx=1)
    initial_values = np.tile(initial_values[:, :, np.newaxis], (1, 1, 150)) # add orientation

    # STEP 3: Time length for computations
    lookback_length = 1.0
    t_step = 0.05

    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # STEP 4: System dynamics for computation
    sys = DubinsCapture(dMax=0.0, uMode="max", dMode="min")

    # STEP 5: Initialize plotting option
    po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1], slicesCut=[50], colorscale="Bluered", save_fig=False, filename="plots/4D_0_sublevel_set", interactive_html=True)

    # STEP 6: Call HJSolver function
    comp_method = { "TargetSetMode": "None"}
    
    settings = {"system": sys, "grid": g, "initial_values": initial_values, "time": tau, "comp_method": comp_method, "plot_options": po}
    return settings

def update_safe_set(settings, initial_values):
    settings["initial_values"] = initial_values
    result = solve(settings)
    return result

def solve(settings):
    sys = settings["system"]
    g = settings["grid"]
    initial_values = settings["initial_values"]
    tau = settings["time"]
    comp_method = settings["comp_method"]
    po = settings["plot_options"]
    result = HJSolver(sys, g, initial_values, tau, comp_method, po, saveAllTimeSteps=True)
    return result

def plot_contour_map(grid_data, vmin=-10, vmax=20, fig=None, ax=None):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    cm = plt.imshow(grid_data, cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(cm)
    plt.title('Contour Map')

if __name__ == "__main__":
    settings = get_initial_settings()
    # po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1], slicesCut=[50], colorscale="Bluered", save_fig=False, filename="plots/4D_0_sublevel_set", interactive_html=True)
    # settings["plot_options"] = po
    result = solve(settings)
    plot_contour_map(result[:, :, 0, -1].T)
    plt.show()
    # for i in range(4):
    #     result = update_safe_set(settings, result[:, :, :, 0])
    #     # plot_contour_map(result[:,:,0,0])
    # po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1], slicesCut=[50], colorscale="Bluered", save_fig=False, filename="plots/4D_0_sublevel_set", interactive_html=True)
    # settings["plot_options"] = po
    # update_safe_set(settings, result[:, :, :, 0])
    # plt.show()


