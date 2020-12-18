import json
import os
from pathlib import Path

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import pybullet_envs  # register pybullet envs from bullet3

from actor_critic_gnn_policy import ActorCriticGNNPolicy

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

model = PPO(ActorCriticGNNPolicy, env, verbose=1, policy_kwargs={
    'features_extractor_kwargs': {'agent_structure': task_log}})
model.learn(total_timesteps=10000)
model.save("a2c_ant")
