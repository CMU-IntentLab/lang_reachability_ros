# Installation
- create a new conda environment exactly equal to lang-reachability (say lang-reachability-ros) just to preserve the previous one in case things go wrong
- install habitat and owl-vit dependencies using the env.yaml file we have
- then install ROS Noetic following this instructions: https://robostack.github.io/GettingStarted.html#__tabbed_4_1
- make sure to remove the line "source /opt/ros/noetic/setup.bash" from your .bashrc if you have it; otherwise it will cause issues now that we are using it with conda
- install the teleoperation package with "conda install ros-noetic-teleop-twist-keyboard"
- create another conda environment (say rtabmap) for rtabmap (Python versions of rtabmap and habitat are incompatible)
- install ROS Noetic in this environment following the same instructions as before
- install rtabmap with conda install ros-noetic-rtabmap-ros

# Running
- in the lang-reachability-ros env.
- in one terminal: run "roscore"
- in another terminal: do not source your workspace as this will break things (and I don't know why, it's just the way life is); run "python3 simulator_node.py". it is very important that you run this with python3 and not rosrun, otherwise it won't be able to import habitat_sim.
- in another terminal: run "rosrun teleop_twist_keyboard teleop_twist_keyboard"
- in another terminal: run "rviz" (for visualization)
- to visualize the map, go to "Add" (bottom left), "Topics" tab, and choose "obstacle point cloud" or something similar
- in the rtabmap env
- run "rosrun rtabmap_slam rtabmap"
