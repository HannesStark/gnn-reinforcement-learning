import os

import pybullet_envs
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import gym
env = gym.make('CartPole-v0')
print(env.action_space)