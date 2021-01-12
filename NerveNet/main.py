import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from NerveNet.models.nerve_net_conv import NerveNetConv

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
checkpoint_callback = CheckpointCallback(save_freq=50, save_path='runs/' + log_name,
                                         name_prefix='rl_model')
model = PPO("GnnPolicy",
            env,
            verbose=1,
            policy_kwargs={
                'mlp_extractor_kwargs': {
                    'task_name': task_name,
                    'xml_assets_path': None,
                },
                'net_arch':  {
                    "input": [
                        (nn.Linear, 6),
                    ],
                    "propagate": [
                        (NerveNetConv, 12),
                        # (NerveNetConv, 12),
                        # (NerveNetConv, 12)
                    ],
                    "policy": [
                        (nn.Linear, 64),
                        (nn.Linear, 64)
                    ],
                    "value": [
                        (nn.Linear, 64),
                        (nn.Linear, 64)
                    ]
                },
                "activation_fn":  nn.Tanh,
            },
            tensorboard_log="runs",
            learning_rate=3.0e-4,
            # batch_size=64,
            # n_epochs=10,
            n_steps=1024)

model.learn(total_timesteps=1000000, callback=checkpoint_callback,
            tb_log_name=log_name)
model.save("a2c_ant")
