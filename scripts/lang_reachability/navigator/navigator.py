import rospy
import torch
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from .planner import mppi
from ..systems.dubins3d import dubins_dynamics_tensor
import tf.transformations
class Navigator:
    def __init__(self, planner_type="mppi", device='cpu', dtype=torch.float32, dt=0.1):

        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.planner_type = planner_type

        self._odom_torch = None
        self.planner = self._start_planner()
        self._map_torch = None  # Initialize later with the map data
        self._cell_size = None  # Initialize later with the map resolution
        self._map_origin_torch = None  # Initialize later with the map origin
        self._goal_torch = None
        self._goal_thresh = 0.1
        print('navigator initialized')


    def get_command(self):
        x = self._odom_torch[0]
        y = self._odom_torch[1]
        dist_goal = torch.sqrt((x - self._goal_torch[0]) ** 2 + (y - self._goal_torch[1]) ** 2)
        if dist_goal.item() < self._goal_thresh:
            return 0, 0
        command = None
        if self.planner_type == "mppi":
            command = self.planner.command(self._odom_torch)
        return command

    def set_odom(self, odom: Odometry):
        self._odom_torch = torch.tensor([odom.pose.pose.position.x,
                                         odom.pose.pose.position.y,
                                         self._quaternion_to_yaw(odom.pose.pose.orientation)],
                                        dtype=self.dtype, device=self.device)

    def set_map(self, map: OccupancyGrid):
        self._map_torch = torch.tensor(map.data, dtype=self.dtype, device=self.device).reshape(map.info.height,
                                                                                               map.info.width)
        self._cell_size = map.info.resolution
        self._map_origin_torch = torch.tensor([map.info.origin.position.x, map.info.origin.position.y],
                                              dtype=self.dtype, device=self.device)

    def set_goal(self, goal: PoseStamped):
        self._goal_torch = torch.tensor([goal.pose.position.x, goal.pose.position.y],
                                        dtype=self.dtype, device=self.device)


    def get_sampled_trajectories(self):
        if self.planner_type == "mppi":
            # states: torch.tensor, shape(M, K, T, nx)
            trajectories = self.planner.states
            M, K, T, nx = trajectories.shape
            return trajectories.view(M*K, T, nx)

    def make_mppi_config(self):
        mppi_config = {}

        mppi_config['dynamics'] = dubins_dynamics_tensor
        mppi_config['running_cost'] = self.mppi_cost_func
        mppi_config['nx'] = 3    # [x, y, theta]
        mppi_config['dt'] = self.dt
        mppi_config['noise_sigma'] = torch.tensor([[0.8, 0], [0, 1]], dtype=self.dtype, device=self.device)
        mppi_config['num_samples'] = 200
        mppi_config['horizon'] = 20
        mppi_config['device'] = self.device
        mppi_config['u_min'] = torch.tensor([0.0, -3.14])
        mppi_config['u_max'] = torch.tensor([4.0, 3.14])
        mppi_config['lambda_'] = 1
        mppi_config['rollout_samples'] = 1
        mppi_config['terminal_state_cost'] = self.mppi_terminal_state_cost_funct
        mppi_config['rollout_var_cost'] = 0.1  # Increase from 0
        mppi_config['rollout_var_discount'] = 0.9  # Adjust from 0.95
        mppi_config['u_init'] = torch.tensor([0.0, 0.0], dtype=self.dtype, device=self.device)

        return mppi_config


    def _start_planner(self,):
        if self.planner_type == 'mppi':
            mppi_config = self.make_mppi_config()
            return mppi.MPPI(**mppi_config)

    def _compute_collision_cost(self, current_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        current_state: shape(num_samples, dim_x)
        action: shape(num_samples, dim_u)

        return:
        cost: shape(num_samples)
        """
        position_map = (current_state[..., :2] - self._map_origin_torch) / self._cell_size
        position_map = torch.round(position_map).long().to(self.device)

        is_out_of_bound = torch.logical_or(
            torch.logical_or(
                position_map[..., 0] < 0, position_map[..., 0] >= self._map_torch.shape[1]
            ),
            torch.logical_or(
                position_map[..., 1] < 0, position_map[..., 1] >= self._map_torch.shape[0]
            ),
        )
        position_map[..., 0] = torch.clamp(position_map[..., 0], 0, self._map_torch.shape[1] - 1)
        position_map[..., 1] = torch.clamp(position_map[..., 1], 0, self._map_torch.shape[0] - 1)
        # Collision check
        collisions = self._map_torch[position_map[..., 1], position_map[..., 0]]
        # Out of bound cost
        collisions[is_out_of_bound] = 1.0
        return collisions

    def mppi_cost_func(self, current_state: torch.Tensor, action: torch.Tensor, t, weights=(1, 100000)) -> torch.Tensor:
        """
        current_state: shape(num_samples, dim_x)
        return:
        cost: torch.tensor, shape(num_samples, 1)
        """
        # print(current_state[:, :2])
        dist_goal_cost = torch.norm(current_state[:, :2] - self._goal_torch, dim=1)
        # print(f'dist_goal_cost: \n{dist_goal_cost}')
        dynamic_collision_weight = weights[1] * torch.exp(torch.tensor([-t], dtype=self.dtype, device=self.device))
        collision_cost = self._compute_collision_cost(current_state, action)

        cost = weights[0] * dist_goal_cost + dynamic_collision_weight * collision_cost
        return cost

    def mppi_terminal_state_cost_funct(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        states: shape(M*K, T, dim_x)
        """
        return self.mppi_cost_func(states, actions, 1, weights=(1, 100000))


    def _quaternion_to_yaw(self, quaternion: Quaternion):
        """
        Convert a quaternion into a yaw angle (theta).

        :param quaternion: Quaternion in the format [x, y, z, w]
        :return: Yaw angle in radians
        """
        euler = tf.transformations.euler_from_quaternion([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w
        ])
        yaw = euler[2]  # Yaw is the third element in the returned tuple
        return yaw


