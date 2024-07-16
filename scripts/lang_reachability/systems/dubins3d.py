import numpy as np

class Dubins3D:
    """ A discrete time dubins car with state
    [x, y, theta] and actions [v, w]. The dynamics are:

    x(t+1) = x(t) + saturate_linear_velocity(v(t)) cos(theta_t)*delta_t
    y(t+1) = y(t) + saturate_linear_velocity(v(t)) sin(theta_t)*delta_t
    theta(t+1) = theta_t + saturate_angular_velocity(w(t))*delta_t
    """

    def __init__(self, vmax=1, wmax=1, dt=0.1, init_x=0.0, init_y=0.0, init_theta=0.0):
        self.dt = dt
        self.vmax = vmax
        self.wmax = wmax

        self.state = np.array([init_x, init_y, init_theta])

    def update_robot_state(self, x, y, theta):
        self.state = np.array([x, y, theta%(2*np.pi)])

    def step(self, v, w):
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        
        x_ = x + self.saturate_linear_velocity(v) * np.cos(theta)*self.dt
        y_ = y + self.saturate_linear_velocity(v) * np.sin(theta)*self.dt
        theta_ = theta + self.saturate_angular_velocity(w) * self.dt
        self.update_robot_state(x_, y_, theta_)

    def saturate_linear_velocity(self, v):
        return np.sign(v) * np.min(np.abs([v, self.vmax]))
    
    def saturate_angular_velocity(self, w):
        return np.sign(w) * np.min(np.abs([w, self.wmax]))

