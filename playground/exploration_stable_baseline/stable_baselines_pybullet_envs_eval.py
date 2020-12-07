import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3

env = gym.make('AntBulletEnv-v0')

model = A2C.load("a2c_ant")


env.render()  # call this before env.reset, if you want a window showing the environment
obs = env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
