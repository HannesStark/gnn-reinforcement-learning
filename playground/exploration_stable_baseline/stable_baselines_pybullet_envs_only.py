
import numpy as np
import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3
import NerveNet.gym_envs.pybullet.register_disability_envs

env = gym.make('AntSixLegsEnv-v0')


env.render()  # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked
action_space = env.action_space
observation_space = env.observation_space

obs = env.reset()
for i in range(5000):
    action = env.action_space.sample()
    no_action = np.zeros_like(action)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
