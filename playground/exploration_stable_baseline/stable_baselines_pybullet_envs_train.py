from datetime import datetime

import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3

env_name = 'AntBulletEnv-v0'

env = gym.make(env_name)

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="runs")
model.learn(total_timesteps=10000, tb_log_name='{}_{}'.format(env_name,datetime.now().strftime('%d-%m_%H-%M-%S')))
model.save("a2c_ant")
