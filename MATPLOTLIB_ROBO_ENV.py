import math
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import true
from td3_matenv import Agent
import tensorflow as tf


show_animation = True


class RobotType(Enum):
    circle = 0
    rectangle = 1


class MatPlotEnv:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.x = np.empty([5])
        self.u = np.empty([2])
        self.goal = np.empty([2])
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = math.pi / 4.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        res = math.pi / 12.0  # [rad/s]
        self.action_space = [-3*res, -2*res, -res, 0*res, 1*res, 2*res, 3*res]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked        
        self.robot_type = RobotType.circle

        if robot_type == RobotType.circle:
            # Also used to check if goal is reached in both types
            self.robot_radius = 0.75  # [m] for collision check

        if robot_type == RobotType.rectangle:
            self.robot_width = 0.5  # [m] for collision check
            self.robot_length = 1.2  # [m] for collision check

        # obstacles [x(m) y(m), ....]
        obs_pos = np.empty((0, 2), dtype=int)

        for i in range(0, int(input("how many obstacles you need-> "))):
            new_obs_posn = np.array(
                [np.random.randint(0, 20), np.random.randint(0, 20)])
            obs_pos = np.append(obs_pos, new_obs_posn.transpose())
        obs_pos = np.reshape(obs_pos, (obs_pos.size//2, 2))

        self.ob = obs_pos
        
        self.dist = np.zeros(self.ob.shape[0]) #LIDAR-ish sensor implemented 

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

    def reset(self):
        # default parameters for the environment
        self.x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
        self.goal = np.array([10, 10])
        # trajectory = np.array(x) #stores track the path covered by the robot then uncomment the variables using trajectory variable
        self.u = np.array([1.0, 0.0])
        return self


def motion(env):
    """
    motion model
    """   
    env.x[0] += env.u[0] * math.cos(env.x[2]) * env.dt  # x
    env.x[1] += env.u[0] * math.sin(env.x[2]) * env.dt  # y
    env.x[2] += env.u[1] * env.dt  # yaw
    env.x[3] = env.u[0]  # linear vel
    env.x[4] = env.u[1]  # angular vel
    return env


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -
                             config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")

    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def get_dist(x1, x2):
    ndist = np.square(x2-x1)
    ndist = np.sqrt(np.abs(np.sum(ndist)))
    return ndist


def get_reward(env, reward, is_collided, is_at_goal,prev_state):
    # our model for rewarding
    # distance comparision from current state and prev state
    # per collison= -10
    curr_dist_from_goal = math.hypot(env.x[0] - env.goal[0], env.x[1] - env.goal[1])

    prev_dist_from_goal = math.hypot(prev_state[0] - env.goal[0], prev_state[1] - env.goal[1])
    if(curr_dist_from_goal>prev_dist_from_goal):
        reward = -5
    else:
        reward = 1
    
    
    if(is_collided):
        is_collided = False
        reward += -10

    # when reaches goal
    if(is_at_goal):
        is_at_goal = True
        reward += 900
    return reward


def env_feedback(env, is_collided, is_at_goal):
    obs = env.ob

    # defining boundary
    if(env.x[0] >= 20 or env.x[0] < 0 or env.x[1] >= 20 or env.x[1] < 0):
        print("outside boundary")
        is_collided = True

    # obstacle collision check/input from all the obstacles
    for i in range(obs.shape[0]):
        ndist = get_dist(obs[i, :], env.x[0:2])
        env.dist[i] = ndist
        if ndist <= env.robot_radius:
            is_collided = True
            print("collided")
            # to update its position after collison
            env.x[0] -= env.u[0] * math.cos(env.x[2]) * env.dt  # x
            env.x[1] -= env.u[0] * math.sin(env.x[2]) * env.dt  # y
            break

    # check reaching goal
    dist_to_goal = math.hypot(env.x[0] - env.goal[0], env.x[1] - env.goal[1])
    if dist_to_goal <= 2*env.robot_radius:
        print("Goal!!")
        is_at_goal = True
    return env, is_collided, is_at_goal

if __name__ == '__main__':
    #Choose which one you would like to use ;)
    
    # main(robot_type=RobotType.rectangle)
    main(robot_type=RobotType.circle)