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

basepath = Path(os.getcwd())
print(basepath)
# make sure your working directory is the repository root.
if basepath.name != "tum-adlr-ws21-04":
    os.chdir(basepath.parent)

basepath = Path(os.getcwd())


graph_logs_dir = basepath / "graph_logs_new"

task_name = 'AntBulletEnv-v0'
nervenet_assets_dir = Path(os.getcwd()).parent / \
    "NerveNet" / "environments" / "assets"


env = gym.make(task_name)

log_name = '{}_{}'.format(task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='runs/' + log_name,
                                         name_prefix='rl_model')
model = A2C("GnnPolicy",
            env,
            # reducing batch_size to 1
            n_steps=8,
            verbose=1,
            policy_kwargs={
                'mlp_extractor_kwargs': {
                    'task_name': task_name,
                    'xml_assets_path': None
                }
            },
            tensorboard_log="runs")

model.learn(total_timesteps=10000, callback=checkpoint_callback,
            tb_log_name=log_name)
model.save("a2c_ant")
