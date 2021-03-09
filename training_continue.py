import argparse
import copy
import json
import os
from typing import Callable

from datetime import datetime
from pathlib import Path

import json
import pyaml
import torch
import yaml
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import MlpPolicy
from torch import nn

import pybullet_data
import pybullet_envs  # register pybullet envs from bullet3

from NerveNet.graph_util.mujoco_parser_settings import ControllerOption, EmbeddingOption, RootRelationOption
from NerveNet.models import nerve_net_conv
from NerveNet.policies import register_policies
import NerveNet.gym_envs.pybullet.register_disability_envs

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from util import LoggingCallback

algorithms = dict(A2C=A2C, PPO=PPO)
activation_functions = dict(Tanh=nn.Tanh, ReLU=nn.ReLU)

controller_option = dict(shared=ControllerOption.SHARED,
                         seperate=ControllerOption.SEPERATE,
                         unified=ControllerOption.UNIFIED)

embedding_option = dict(shared=EmbeddingOption.SHARED,
                        unified=EmbeddingOption.UNIFIED)

root_option = dict(none=RootRelationOption.NONE,
                   body=RootRelationOption.BODY,
                   unified=RootRelationOption.ALL)


def train(args):
    cuda_availability = torch.cuda.is_available()
    print('\n*************************')
    print('`CUDA` available: {}'.format(cuda_availability))
    print('Device specified: {}'.format(args.device))
    print('*************************\n')

    # load the config of the trained model:
    with open(args.pretrained_output / "train_arguments.yaml") as yaml_data:
        pretrain_arguments = yaml.load(yaml_data,
                                       Loader=yaml.FullLoader)

    pretrained_model = algorithms[pretrain_arguments["alg"]].load(
        args.pretrained_output / "".join(pretrain_arguments["model_name"].split(".")[:-1]), device='cpu')

    # Prepare tensorboard logging
    log_name = '{}_{}'.format(
        pretrain_arguments["experiment_name"],  datetime.now().strftime('%d-%m_%H-%M-%S'))
    run_dir = args.tensorboard_log + "/" + log_name
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    callbacks = []
    # callbacks.append(CheckpointCallback(
    #    save_freq=1000000, save_path=run_dir, name_prefix='rl_model'))
    callbacks.append(LoggingCallback(logpath=run_dir))

    train_args = copy.copy(pretrain_arguments)
    pyaml.dump(train_args, open(
        os.path.join(run_dir, 'train_arguments.yaml'), 'w'))

    # Create the vectorized environment
    n_envs = pretrain_arguments["n_envs"]  # Number of processes to use
    env = make_vec_env(pretrain_arguments["task_name"], n_envs=n_envs)

    pretrained_model.env = env
    pretrained_model.learn(total_timesteps=args.total_timesteps,
                           callback=callbacks,
                           tb_log_name=log_name)

    pretrained_model.save(os.path.join(args.tensorboard_log +
                                       "/" + log_name, args.model_name))


def dir_path(path):
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained_output',
                   help="The directory where the pretrained output & configs were logged to",
                   type=dir_path,
                   default='runs/GNN_PPO_inp_32_pro_32164_pol_16_val_64_64_N2048_B512_lr2e-04_GNNValue_0_EmbOpt_shared_mode_action_per_controller_Epochs_10_Nenvs_8_GRU_AntBulletEnv-v0_09-03_18-00-53')

    p.add_argument("--total_timesteps",
                   help="The total number of samples (env steps) to train on",
                   type=int,
                   default=4000000)
    p.add_argument('--tensorboard_log',
                   help='the log location for tensorboard (if None, no logging)',
                   default="runs")
    p.add_argument('--n_envs',
                   help="Number of environments to run in parallel to collect rollout. Each environment requires one CPU",
                   type=int,
                   default=8)

    p.add_argument('--device',
                   help='Device (cpu, cuda, ...) on which the code should be run.'
                        'Setting it to auto, the code will be run on the GPU if possible.',
                   default="auto")

    p.add_argument('--experiment_name',
                   help='name to append to the tensorboard logs directory',
                   default=None)
    p.add_argument('--experiment_name_suffix',
                   help='name to append to the tensorboard logs directory',
                   default=None)

    p.add_argument('--model_name',
                   help='The name of your saved model',
                   default='model.zip')

    args = p.parse_args()

    return args


if __name__ == '__main__':
    train(parse_arguments())
