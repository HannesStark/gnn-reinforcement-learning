import argparse
import copy
import json
import os

from datetime import datetime
from pathlib import Path

import json
import pyaml
import torch
import yaml
import numpy as np

from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

import pybullet_envs  # register pybullet envs from bullet3

import NerveNet.gym_envs.pybullet.register_disability_envs

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from util import LoggingCallback
algorithms = dict(A2C=A2C, PPO=PPO)


def init_evaluate(args):

    # load the config of the trained model:
    with open(args.train_output / "train_arguments.yaml") as yaml_data:
        train_arguments = yaml.load(yaml_data,
                                    Loader=yaml.FullLoader)

    model = algorithms[train_arguments["alg"]].load(
        args.train_output / args.model_name, device='cpu')
    env_name = train_arguments["task_name"]

    env = gym.make(env_name)

    if args.render:
        env.render()  # call this before env.reset, if you want a window showing the environment

    def logging_callback(local_args, globals):
        if local_args["done"]:
            i = len(local_args["episode_rewards"])
            episode_reward = local_args["episode_reward"]
            episode_length = local_args["episode_length"]
            print(f"Finished {i} episode with reward {episode_reward}")

    episode_rewards, episode_lengths = evaluate_policy(model,
                                                       env,
                                                       n_eval_episodes=args.num_episodes,
                                                       render=args.render,
                                                       deterministic=True,
                                                       return_episode_rewards=True,
                                                       callback=logging_callback)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"mean_length:{mean_length:.2f} +/- {std_length:.2f}")

    eval_dir = args.train_output / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    np.save(eval_dir / "episode_rewards.npy", episode_rewards)
    np.save(eval_dir / "episode_lengths.npy", episode_lengths)


def dir_path(path):
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'))

    p.add_argument('--train_output',
                   help="The directory where the training output & configs were logged to",
                   type=dir_path,
                   default='runs/MLP_PPO_pi64_64_vf64_64_N2048_B64_lr2e-04_GNNValue_0_EmbOpt_shared_AntBulletEnv-v0_02-03_10-45-07')

    p.add_argument("--num_episodes",
                   help="The number of episodes to run to evaluate the model",
                   type=int,
                   default=20)

    p.add_argument('--model_name',
                   help='The name of your saved model',
                   default='model.zip')

    p.add_argument('--render',
                   help='Whether to render the evaluation with pybullet client',
                   type=bool,
                   default=False)

    args = p.parse_args()

    if args.config is not None:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list) and arg_dict[key] is not None:
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    return args


if __name__ == '__main__':
    init_evaluate(parse_arguments())
