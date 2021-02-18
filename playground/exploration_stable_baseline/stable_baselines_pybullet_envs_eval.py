import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from pathlib import Path

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import pybullet_envs  # register pybullet envs from bullet3
import NerveNet.gym_envs.pybullet.register_disability_envs
import time

model = PPO.load("model.zip", device='cpu')
env_name = 'AntBulletEnv-v0'
model_dir = Path(
    "C:\\Users\\tsbau\\git\\tum-adlr-ws21-04\\runs\\MLP_S64_P64_V64_N1000_B64_lr3e-4_AntSixLegsEnv-v0_17-02_00-46-47")

model = PPO.load(model_dir / "model.zip", device='cpu')
env_name = 'AntSixLegsEnv-v0'

env = gym.make(env_name)

env.render()  # call this before env.reset, if you want a window showing the environment


def evaluate(model, num_episodes=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment

    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            time.sleep(0.01)
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            env.render('human')
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


mean_reward = evaluate(model, num_episodes=1)
