import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3

env = gym.make('AntBulletEnv-v0')


env.render()  # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked


obs = env.reset()
for i in range(500):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
