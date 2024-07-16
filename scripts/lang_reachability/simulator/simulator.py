import numpy as np
import os, sys
import habitat_sim
import json
# from habitat.utils.visualizations import maps
from scipy.spatial.transform import Rotation
from pathlib import Path
import matplotlib.pyplot as plt
dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH


# TODO:
# - function that converts from simulator default frame to our world frame where x = forward, y = right, z = up

class Simulator:
    def __init__(self, dataset_name, test_scene, sim_settings=None) -> None:
        with open(os.path.join(dir_path, 'configs/datasets', f'{dataset_name}.json')) as f:
            self.dataset_settings = json.load(f)
        self.agent_sensor_settings = self.dataset_settings['agent_sensor_settings']
        self.dataset_2_hab_settings = self.dataset_settings['dataset_2_hab_settings']
        self.agent_sensor_settings['scene'] = test_scene
        self.cfg = None
        self.top_down_map = None
        if sim_settings is None:
            if dataset_name == 'hssd':
                self.cfg = self._make_config()
            elif dataset_name == 'hm3d':
                self.cfg = self._make_config()
            elif dataset_name == 'hssd-hab':
                self.cfg = self._make_config_hssd_hab()
            self.test_scene = test_scene
        else:
            self.agent_sensor_settings = sim_settings
            self.test_scene = sim_settings["scene"]
        
        self.init_r = np.pi/2
        self.init_p = np.pi/2
        self.init_y = np.pi/2

        self.dataset_name = dataset_name
        self.sim = self._init_sim(self.cfg)
        self.agent = self._init_agent(self.agent_sensor_settings)
        # self._calculate_navmesh()
        # self.get_top_down_map()

    def _calculate_navmesh(self):
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_params = self.dataset_settings['navmesh_settings']
        self.sim.navmesh_visualization = False
        navmesh_settings.set_defaults()
        navmesh_settings.cell_size = navmesh_params["cell_size"]
        navmesh_settings.cell_height = navmesh_params["cell_height"]
        navmesh_settings.agent_height = navmesh_params["agent_height"]
        navmesh_settings.agent_radius = navmesh_params["agent_radius"]
        navmesh_settings.agent_max_climb = navmesh_params["agent_max_climb"]
        navmesh_settings.agent_max_slope = navmesh_params["agent_max_slope"]
        navmesh_settings.filter_low_hanging_obstacles = navmesh_params["filter_low_hanging_obstacles"]
        navmesh_settings.filter_ledge_spans = navmesh_params["filter_ledge_spans"]
        navmesh_settings.filter_walkable_low_height_spans = navmesh_params["filter_walkable_low_height_spans"]
        navmesh_settings.region_min_size = navmesh_params["region_min_size"]
        navmesh_settings.region_merge_size = navmesh_params["region_merge_size"]
        navmesh_settings.edge_max_len = navmesh_params["edge_max_len"]
        navmesh_settings.edge_max_error = navmesh_params["edge_max_error"]
        navmesh_settings.verts_per_poly = navmesh_params["verts_per_poly"]
        navmesh_settings.detail_sample_dist = navmesh_params["detail_sample_dist"]
        navmesh_settings.detail_sample_max_error = navmesh_params["detail_sample_max_error"]
        navmesh_settings.include_static_objects = True

        self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        return
    
    def _make_config_hssd_hab(self):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = os.path.join(
            dir_path,
            'data/hssd-hab/hssd-hab.scene_dataset_config.json'
        )
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        sim_cfg.scene_id = self.agent_sensor_settings["scene"]

        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.agent_sensor_settings['sensor_resolution']
        # init pose and ori for hssd #
        rgb_sensor_spec.position = self.agent_sensor_settings['sensor_position_to_agent']
        rgb_sensor_spec.orientation = self.agent_sensor_settings['sensor_orientation_to_agent']

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.agent_sensor_settings['sensor_resolution']
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        # init pose and ori for hssd #
        depth_sensor_spec.position = self.agent_sensor_settings['sensor_position_to_agent']
        depth_sensor_spec.orientation = self.agent_sensor_settings['sensor_orientation_to_agent']

        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        print(f'camera orientation: {rgb_sensor_spec.orientation}')
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def _make_config(self):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.agent_sensor_settings["scene"]

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # attach only a RGB visual sensor to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.agent_sensor_settings['sensor_resolution']
        # init pose and ori for hssd #
        rgb_sensor_spec.position = self.agent_sensor_settings['sensor_position_to_agent']
        rgb_sensor_spec.orientation = self.agent_sensor_settings['sensor_orientation_to_agent']

        # attach a depth sensor to the agent
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.agent_sensor_settings['sensor_resolution']
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        # init pose and ori for hssd #
        depth_sensor_spec.position = self.agent_sensor_settings['sensor_position_to_agent']
        depth_sensor_spec.orientation = self.agent_sensor_settings['sensor_orientation_to_agent']

        self.sensors_specs = {'camera': rgb_sensor_spec, 'depth': depth_sensor_spec}

        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        # print(f'camera orientation: {rgb_sensor_spec.orientation}')
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def _init_sim(self, cfg):
        return habitat_sim.Simulator(cfg)
    
    def _init_agent(self, settings, init_xyz=[0.0, 0.0, 0.0], init_rpy=[0.0, 0.0, 0.0]) -> habitat_sim.Agent:
        # initialize an agent
        agent = self.sim.initialize_agent(settings["default_agent"])

        # set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(init_xyz)  # in world space
        agent_state.rotation = np.array(Rotation.from_euler('xyz', init_rpy, degrees=False).as_quat())
        agent.set_state(agent_state)

        # get agent state
        agent_state = agent.get_state()
        print("initialized agent at:\n\tagent_state: position", agent_state.position, "\n\trotation", agent_state.rotation)

        return agent

    def get_agent_state(self):
        """Returns the agent state in the world frame"""
        state_sim = self.agent.get_state()
        pos = state_sim.position
        ori = Rotation.from_quat(state_sim.rotation.components).as_euler("xyz", degrees=False)

    def get_inv_camera_extrinsics_mat(self, robot_state):
        """
        returns camera -> world transformation matrix, aka the inverse of the camera extrinsics matrix
        """
        T = self.get_camera_extrinsics_mat(robot_state)
        return np.linalg.inv(T)

    def get_camera_extrinsics_mat(self, robot_state):
        """
        returns world -> camera transformation matrix, aka the camera extrinsics matrix
        """
        # get agents state in world frame
        x = robot_state[0] + self.agent_sensor_settings["sensor_position_to_agent"][0]
        y = robot_state[1] + self.agent_sensor_settings["sensor_position_to_agent"][1]
        # z = self.agent_sensor_settings["sensor_position_to_agent"][2]
        roll = -np.pi/2
        yaw = robot_state[2] - np.pi/2
        # compute rotation and translation of the camera relative to world frame
        rot_roll = np.array([[1,             0,            0],
                             [0, np.cos(roll),  np.sin(roll)],
                             [0, -np.sin(roll), np.cos(roll)]])
        rot_yaw = np.array([[np.cos(yaw),  np.sin(yaw), 0],
                            [-np.sin(yaw), np.cos(yaw), 0],
                            [0,            0,           1]])
        rot = np.matmul(rot_roll, rot_yaw)
        t = np.array([x, y, 1])
        t_inv = np.matmul(rot, -t)
        # concatenate rotation and translation to build homogenenous transformation matrix
        mat = np.hstack((rot, np.c_[t_inv]))
        mat = np.vstack((mat, np.array([0, 0, 0, 1])))
        return mat

    def get_camera_intrinsics_mat(self):
        """Returns the camera intrinsics matrix"""
        hfov = np.deg2rad(float(self.sensors_specs['camera'].hfov))
        height, width = self.sensors_specs['camera'].resolution
        f = width/(2*np.tan(hfov/2))
        mat = np.array([[f,  0., width/2,],
                        [0., f,  height/2,],
                        [0., 0., 1.,]])
        return mat

    def get_inv_camera_intrinsics_mat(self):
        """Returns the inverse of the camera intrinsics matrix"""
        hfov = np.deg2rad(float(self.sensors_specs['camera'].hfov))
        height, width = self.sensors_specs['camera'].resolution
        f = width/2 * (1/np.tan(hfov/2))
        mat = np.array([[1/f,  0., -width/(2*f),],
                        [0., 1/f,  -height/(2*f),],
                        [0., 0., 1.,]])
        return mat

    def get_camera_projection_mat(self):
        K = self.get_camera_intrinsics_mat()
        T = self.get_camera_extrinsics_mat()
        return K @ T

    def get_observation(self, x, y, theta):
        # hssd's frame convention #
        xyz = x * np.array(self.dataset_2_hab_settings['dx_2_hx']) + y * np.array(self.dataset_2_hab_settings['dy_2_hy'])
        rpy = theta * np.array(self.dataset_2_hab_settings['dtheta_2_htheta'])
        self.update_agent_state(xyz, rpy)
        observations = self.sim.get_sensor_observations()
        return observations
    
    def get_rgb_observation(self, x, y, theta):
        observations = self.get_observation(x, y, theta)
        return observations["color_sensor"]
    
    def get_depth_observation(self, x, y, theta):
        observations = self.get_observation(x, y, theta)
        return observations["depth_sensor"]

    # def get_top_down_map(self, height=0.0):
    #     self.top_down_map = maps.get_topdown_map(
    #         self.sim.pathfinder, map_resolution=1024, height=height
    #     )
    #     # 0: occupied, 1: not occupied, 2, border
    #     return

    def save_top_down_map(self, path):
        recolor_map = np.array(
            [[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[self.top_down_map]
        plt.imshow(top_down_map)
        plt.title("top_down_map.png")
        plt.savefig(path)

    def world_to_pixel(self, x, y, z, robot_state):
        K = self.get_camera_intrinsics_mat()
        T = self.get_camera_extrinsics_mat(robot_state=robot_state)
        pw = np.array([x, y, z, 1])
        pi = np.matmul(T, pw)
        pp = np.matmul(K, pi[:3])
        return pp[0]/pp[2], pp[1]/pp[2]

    def pixel_to_world(self, uv, robot_state):
        K_inv = self.get_inv_camera_intrinsics_mat()
        T_inv = self.get_inv_camera_extrinsics_mat(robot_state=robot_state)
        # normalize pixel coordinates
        xy_img = np.matmul(K_inv, uv)
        # transform to world frame
        xy_img = np.vstack((xy_img, np.ones(xy_img.shape[1])))
        xyz = np.matmul(T_inv, xy_img)
        x = xyz[0]
        y = xyz[1]
        return x, y

    def estimate_object_position(self, robot_state, depth, bbox, threshold=1.2):
        # get indexes of bounding box pixels
        try:
            bbox = bbox.cpu().numpy()
        except AttributeError as e: # just in case it is not a torch tensor
            pass

        height, width = self.sensors_specs['camera'].resolution
        u_lim = np.clip([bbox[0], bbox[2]], 0, width).astype(int)
        v_lim = np.clip([bbox[1], bbox[3]], 0, height).astype(int)
        u, v = np.meshgrid(np.arange(u_lim[0], u_lim[1]), np.arange(v_lim[0], v_lim[1]))
        u = u.flatten()
        v = v.flatten()
        # get depth for each pixel in the bounding box
        depth_obj = depth[v_lim[0]:v_lim[1], u_lim[0]:u_lim[1]]
        depth_x = depth_obj.flatten()
        depth_y = depth_obj.T.flatten()
        # add depth information
        u = depth_x * u
        v = depth_y * v
        ones = depth_x * np.ones(u.shape).flatten()
        # only consider pixels below depth threshold
        iu = np.where(depth_x < threshold)[0]
        iv = np.where(depth_y < threshold)[0]
        uv = np.vstack((u[iu], v[iv], ones[iu]))
        # pixel -> world
        x, y = self.pixel_to_world(uv, robot_state)
        return x, y

    def update_agent_state(self, xyz, rpy):
        new_state = habitat_sim.AgentState()
        new_state.position = np.array(xyz)  # in world space
        new_state.rotation = np.array(Rotation.from_euler('xyz', rpy, degrees=False).as_quat())
        # new_state.rotation = np.array(self._rpy2quat(init_rpy))
        self.agent.set_state(new_state)