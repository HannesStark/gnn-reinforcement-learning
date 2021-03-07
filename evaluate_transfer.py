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

import pybullet_data
import pybullet_envs  # register pybullet envs from bullet3

from NerveNet.policies import register_policies
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

    with open(args.train_output / "net_arch.txt") as json_data:
        json_str = json_data.read()
        # replace ' with " as a workaround because we didn't do a
        # proper json.dump create the net_arch.txt
        net_arch = json.loads(json_str.replace("'", '"'))

    alg_class = algorithms[train_arguments["alg"]]

    model_old = alg_class.load(
        args.train_output / train_arguments["model_name"], device='cpu')

    env_name = args.transfer_env
    env = gym.make(env_name)

    alg_kwargs = dict()
    policy_kwargs = dict()
    policy_kwargs['base_policy'] = model_old.policy
    policy_kwargs['net_arch'] = net_arch
    policy_kwargs['base_env_task_name'] = train_arguments["task_name"]
    policy_kwargs['base_env_xml_assets_path'] = Path(
        train_arguments["xml_assets_path"])

    # if the base environment was trained on a another system, this path might be wrong.
    # we can't easily fix this in general...
    # but in case it is just the default path to the pybullet_data we can
    base_xml_path_parts = policy_kwargs['base_env_xml_assets_path'].parents._parts
    if "pybullet_data" in base_xml_path_parts:
        policy_kwargs['base_env_xml_assets_path'] = Path(
            pybullet_data.getDataPath()) / "mjcf"
    # also in case it is relative to the repository's root we can:
    if "tum-adlr-ws21-04" in base_xml_path_parts:
        relative_parts_offset = base_xml_path_parts.index("tum-adlr-ws21-04")
        relative_parts = base_xml_path_parts[relative_parts_offset:]
        # assuming the working directory is the tum-adlr-ws21-04 repository root
        policy_kwargs['base_env_xml_assets_path'] = Path(
            os.getcwd()) / "/".join(relative_parts)

    policy_kwargs['transfer_env_task_name'] = args.transfer_env
    policy_kwargs['transfer_env_xml_assets_path'] = args.xml_assets_path

    if "activation_fn" in train_arguments:
        if train_arguments["activation_fn"] is not None:
            policy_kwargs["activation_fn"] = activation_functions[train_arguments["activation_fn"]]

    if "policy" in train_arguments:
        if train_arguments["policy"] == "GnnPolicy":
            policy_kwargs["mlp_extractor_kwargs"] = {
                "task_name": train_arguments["task_name"],
                'device': train_arguments["device"],
                'gnn_for_values': train_arguments["gnn_for_values"],
                'embedding_option': embedding_option[train_arguments["embedding_option"]],
                'xml_assets_path': train_arguments["xml_assets_path"],
            }

    model_transfer = alg_class("MLPTransferPolicy",  # train_arguments["policy"],
                               env,
                               verbose=1,
                               n_steps=train_arguments["n_steps"],
                               policy_kwargs=policy_kwargs,
                               device=train_arguments["device"],
                               tensorboard_log=train_arguments["tensorboard_log"],
                               learning_rate=train_arguments["learning_rate"],
                               batch_size=train_arguments["batch_size"],
                               n_epochs=train_arguments["n_epochs"],
                               **alg_kwargs)

    if args.render:
        env.render()  # call this before env.reset, if you want a window showing the environment

    def logging_callback(local_args, globals):
        if local_args["done"]:
            i = len(local_args["episode_rewards"])
            episode_reward = local_args["episode_reward"]
            episode_length = local_args["episode_length"]
            print(f"Finished {i} episode with reward {episode_reward}")

    episode_rewards, episode_lengths = evaluate_policy(model_transfer,
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
                   default=10)

    p.add_argument("--transfer_env",
                   help="The environment the model should be transfered to",
                   type=str,
                   default="AntSixLegsEnv-v0")

    p.add_argument('--xml_assets_path',
                   help="The path to the directory where the xml of the transfer task's robot is defined",
                   type=dir_path,
                   # default=Path(pybullet_data.getDataPath()) / "mjcf")
                   default=Path(os.getcwd()) / "NerveNet/gym_envs/assets")

    p.add_argument('--render',
                   help='Whether to render the evaluation with pybullet client',
                   type=bool,
                   default=True)

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
