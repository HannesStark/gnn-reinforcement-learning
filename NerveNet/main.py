import json
import os
from pathlib import Path

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import pybullet_envs  # register pybullet envs from bullet3

from NerveNet.actor_critic_gnn_policy import ActorCriticGNNPolicy

basepath = Path(os.getcwd())
print(basepath)
# make sure your working directory is the repository root.
if basepath.name != "tum-adlr-ws21-04":
    os.chdir(basepath.parent)

basepath = Path(os.getcwd())

# %%

graph_logs_dir = basepath / "graph_logs_new"
graph_logs_dir.exists(), graph_logs_dir

task_name = 'AntBulletEnv-v0'

env = gym.make(task_name)

with open(str(graph_logs_dir / f"{task_name}.json")) as json_file:
    task_log = json.load(json_file)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')
model = PPO(ActorCriticGNNPolicy, env, verbose=1, policy_kwargs={
    'features_extractor_kwargs': {'env': env, 'agent_structure': task_log}})
model.learn(total_timesteps=10000, callback=checkpoint_callback)
model.save("a2c_ant")
