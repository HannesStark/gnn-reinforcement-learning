import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import pybullet_envs  # register pybullet envs from bullet3

from ActorCriticGNNPolicy import ActorCriticGNNPolicy

env = gym.make('AntBulletEnv-v0')

model = PPO(ActorCriticGNNPolicy, env, verbose=1, policy_kwargs={"env"})
model.learn(total_timesteps=10000)
model.save("a2c_ant")