"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import pygame
import numpy as np


def move(points, dx, dy):
    return points + np.array([dx, dy])

def rotate(points, center, angle):
    angle_rad = angle
    rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                               [math.sin(angle_rad), math.cos(angle_rad)]])
    points = points - center
    rotated_points = np.dot(points, rotation_matrix)
    rotated_points = rotated_points + center
    return rotated_points

def surf_to_ndarray(surf:pygame.Surface):
    """pygame image to cv2 form image"""
    frame_np = np.array(pygame.surfarray.pixels3d(surf))
    frame_np = np.transpose(frame_np, (1, 0, 2))
    return frame_np

class CartPoleEnv:
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self, for_ddpg=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.surface: pygame.Surface = None
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.for_ddpg = for_ddpg
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        np.random.seed(seed)
        

    def step(self, action):
        if not self.for_ddpg:
            if action not in (0, 1):
                err_msg = "%r (%s) invalid, should be type(int) and in [0, 1]" % (action, type(action))
                raise Exception(err_msg)
            action = [-1, 1][action]
        if action > 1.0 or action < -1.0 :
            err_msg = "%r (%s) invalid, should in [-1, 1]" % (action, type(action))
            raise Exception(err_msg)
        if self.state is None:
            raise Exception("Please run .reset() before .step()") 
        if self.for_ddpg:
            # action = -1 if action < 0 else 1
            action = float(action)
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        """重置环境"""
        np.random.randn()
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self) -> np.ndarray:
        '''渲染当前帧画面'''
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None
        x = self.state[0]
        angle = self.state[2]
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART

        if self.surface is None:
            self.surface = pygame.Surface((screen_width, screen_height))
        
        self.surface.fill("white")
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart_points = np.array([(l, b), (l, t), (r, t), (r, b)])
        cart_points = move(cart_points, cartx, carty)
        cart_color = (0, 0, 0)  # 设置颜色为红色

        # 绘制杆子
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_points = np.array([(l, b), (l, t), (r, t), (r, b)])
        pole_points = move(pole_points, 0, axleoffset)
        pole_points = move(pole_points, cartx, carty)
        pole_color = (255*.8, 255*.6, 255*.4)

        # 绘制旋转中心
        axle_point = np.array([0, 0])
        axle_point = move(axle_point, 0, axleoffset)
        axle_point = move(axle_point, cartx, carty)
        axle_color = (255*.5, 255*.5, 255*.8)
        pole_points = rotate(pole_points, axle_point, angle)
        # 绘制导轨
        track_color = (0, 0, 0)
        pygame.draw.polygon(self.surface, cart_color, cart_points)
        pygame.draw.polygon(self.surface, pole_color, pole_points)
        pygame.draw.circle(self.surface, axle_color, axle_point, polewidth/2)
        pygame.draw.line(self.surface, track_color, (0, carty), (screen_width, carty))
        self.surface = pygame.transform.flip(self.surface, False, True)
        return surf_to_ndarray(self.surface)

    def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        return None

if __name__ == '__main__':
    ... 