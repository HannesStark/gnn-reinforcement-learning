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
from stable_baselines3.common.evaluation import evaluate_policy
import pybullet_envs  # register pybullet envs from bullet3
import NerveNet.gym_envs.pybullet.register_disability_envs  # register custom envs
import time

model_dir = Path(
    "C:\\Users\\tsbau\\git\\tum-adlr-ws21-04\\runs\\MLP_S64_P64_V64_N1000_B64_lr3e-4_AntCpLeftBackBulletEnv-v0_14-02_23-42-32")

model = PPO.load(model_dir / "model.zip", device='cpu')
env_name = 'AntCpLeftBackBulletEnv-v0'

eval_env = gym.make(env_name)

eval_env.render()  # call this before env.reset, if you want a window showing the environment


def logging_callback(local_args, globals):
    if local_args["done"]:
        i = len(local_args["episode_rewards"])
        episode_reward = local_args["episode_reward"]
        episode_length = local_args["episode_length"]
        print(f"Finished {i} episode with reward {episode_reward}")


mean_reward, std_reward = evaluate_policy(model,
                                          eval_env,
                                          n_eval_episodes=10,
                                          render=True,
                                          deterministic=True,
                                          return_episode_rewards=False,
                                          callback=logging_callback)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
