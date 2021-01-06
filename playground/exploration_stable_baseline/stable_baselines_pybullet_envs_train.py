from datetime import datetime
import numpy as np
import gym
from gym import wrappers
from stable_baselines3 import A2C, PPO
import pybullet_envs  # register pybullet envs from bullet3
import json
import os
from datetime import datetime
from pathlib import Path

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import pybullet_envs  # register pybullet envs from bullet3

from NerveNet.policies import register_policies


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


task_name = 'AntBulletEnv-v0'

env = gym.make(task_name)

# model = PPO("MlpPolicy",
#            env,
#            # reducing batch_size to 1
#            n_steps=1024,
#            verbose=1,
#            tensorboard_log="runs", batch_size=32,
#            learning_rate=2.5e-4,
#            gamma=0.99,
#            gae_lambda=0.95,
#            clip_range=0.2,
#            vf_coef=0.5)
#model = PPO("GnnPolicy",
#            env,
#            # reducing batch_size to 1
#            n_steps=1024,
#            verbose=1,
#            tensorboard_log="runs", batch_size=32,
#            learning_rate=1e-3,
#            gamma=0.99,
#            gae_lambda=0.95,
#            clip_range=0.2,
#            vf_coef=0.5,
#            policy_kwargs={
#                'mlp_extractor_kwargs': {
#                    'task_name': task_name,
#                    'xml_assets_path': None
#                }
#            },
#            )
model = PPO("GnnPolicy",
            env,
            # reducing batch_size to 1
            n_steps=2050,
            verbose=1,
            tensorboard_log="runs", batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=5.0,
            vf_coef=0.5,
            policy_kwargs={
                'mlp_extractor_kwargs': {
                    'task_name': task_name,
                    'xml_assets_path': None
                }
            },
            )
mean_reward_before_train = evaluate(model, num_episodes=4)
model.learn(total_timesteps=2000000, tb_log_name='{}_{}'.format(task_name, datetime.now().strftime('%d-%m_%H-%M-%S')))
model.save("a2c_ant")
mean_reward = evaluate(model, num_episodes=4)
print(mean_reward_before_train)
print(mean_reward)
