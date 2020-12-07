import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3

env = gym.make('AntBulletEnv-v0')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("a2c_ant")
