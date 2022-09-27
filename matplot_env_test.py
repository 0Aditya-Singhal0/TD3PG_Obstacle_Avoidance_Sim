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
        # self.max_speed = 1.0  # [m/s]
        # self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = math.pi / 4.0  # [rad/s]
        # self.max_accel = 0.2  # [m/ss]
        # self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        # self.v_resolution = 0.01  # [m/s]
        res = math.pi / 12.0  # [rad/s]
        self.action_space = [-3*res, -2*res, -res, 0*res, 1*res, 2*res, 3*res]
        self.dt = 0.1  # [s] Time tick for motion prediction
        # self.predict_time = 3.0  # [s]
        # self.to_goal_cost_gain = 0.15
        # self.speed_cost_gain = 1.0
        # self.obstacle_cost_gain = 1.0
        # self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked        
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.75  # [m] for collision check

        # if robot_type == RobotType.rectangle
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
        self.dist = np.zeros(self.ob.shape[0])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

    def reset(self):
        # default parameters for the environment put here
        self.x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
        self.goal = np.array([10, 10])
        # trajectory = np.array(x)
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


def main(robot_type=RobotType.circle):
    env = MatPlotEnv()
    env.robot_type = robot_type
    state_shape= np.hstack((env.x,env.dist)).shape
    agent = Agent(alpha=0.001, beta=0.001, input_dims=state_shape, env=env, tau=0.005, batch_size=100, layer1_size=400, layer2_size=300, n_actions=7,warmup=1)
    agent.load_models()
    
    epoch_num = 10000

    is_collided = False
    is_at_goal = False
    score_history = []
    for i in range(epoch_num):
        env = env.reset()
        observation = np.hstack((env.x,env.dist))
        is_collided = False
        is_at_goal = False
        score = -1000
        time = 0
        reward = 0
        while not is_collided and not is_at_goal:
            action = agent.choose_action(observation)
            # observation_, reward, done, info = env.step(action)
            env.u[1] = env.action_space[tf.math.argmax(action,1)[0]]

            env = motion(env)  # simulate robot
            time += env.dt
            observation_ = np.hstack((env.x,env.dist))

            env, is_collided, is_at_goal = env_feedback(env, is_collided, is_at_goal)
            reward = get_reward(env, reward, is_collided, is_at_goal,observation)

            if (time >= 20):  # gives it about 200 steps
                print("Took way too long")
                # reward += -10  # toll for just looping around
                is_collided = True
            score += reward

            observation = observation_

            if show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
                    exit(0) if event.key == 'escape' else None])
                # plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
                plt.plot(env.x[0], env.x[1], "xr")
                plt.plot(env.goal[0], env.goal[1], "xb")
                plt.plot(env.ob[:, 0], env.ob[:, 1], "ok")
                plot_robot(env.x[0], env.x[1], env.x[2], env)
                plot_arrow(env.x[0], env.x[1], env.x[2])
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.0001)

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    print("Done")
    # if show_animation:
    #     plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
    #     plt.pause(0.0001)
    # plt.show()


if __name__ == '__main__':
    # main(robot_type=RobotType.rectangle)
    main(robot_type=RobotType.circle)