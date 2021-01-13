import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3
from stable_baselines3.common.vec_env import unwrap_vec_normalize

env = gym.make('AntBulletEnv-v0')

env.render()  # call this before env.reset, if you want a window showing the environment
obs = env.reset()
for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs)
    env.render()
    if done:
        obs = env.reset()