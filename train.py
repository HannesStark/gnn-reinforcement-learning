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

    # Prepare tensorboard logging
    log_name = '{}_{}_{}'.format(
        args.experiment_name, args.task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
    run_dir = args.tensorboard_log + "/" + log_name
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    callbacks = []
    # callbacks.append(CheckpointCallback(
    #    save_freq=1000000, save_path=run_dir, name_prefix='rl_model'))
    callbacks.append(LoggingCallback(logpath=run_dir))

    train_args = copy.copy(args)
    train_args.config = train_args.config.name
    pyaml.dump(train_args.__dict__, open(
        os.path.join(run_dir, 'train_arguments.yaml'), 'w'))

    # Create the vectorized environment
    n_envs = train_args.n_envs  # Number of processes to use
    env = make_vec_env(args.task_name, n_envs=n_envs)

    # define network architecture
    if "GnnPolicy" in args.policy and args.net_arch is not None:
        for net_arch_part in args.net_arch.keys():
            for i, (layer_class_name, layer_size) in enumerate(args.net_arch[net_arch_part]):
                if hasattr(nn, layer_class_name):
                    args.net_arch[net_arch_part][i] = (
                        getattr(nn, layer_class_name), layer_size)
                elif hasattr(nerve_net_conv, layer_class_name):
                    args.net_arch[net_arch_part][i] = (
                        getattr(nerve_net_conv, layer_class_name), layer_size)
                else:
                    def get_class(x):
                        return globals()[x]

                    c = get_class(layer_size)
                    assert c is not None, f"Unkown layer class '{layer_class_name}'"
                    args.net_arch[net_arch_part][i] = (c, layer_size)

    with open(os.path.join(run_dir, 'net_arch.txt'), 'w') as fp:
        fp.write(str(args.net_arch))

    # Create the model
    alg_class = algorithms[args.alg]
    policy_kwargs = dict()
    if args.net_arch is not None:
        policy_kwargs['net_arch'] = args.net_arch
    if args.activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_functions[args.activation_fn]
    # policy_kwargs['device'] = args.device if args.device is not None else get_device('auto')
    if "GnnPolicy" in args.policy:
        policy_kwargs["mlp_extractor_kwargs"] = {
            "task_name": args.task_name,
            'device': args.device,
            'gnn_for_values': args.gnn_for_values,
            'controller_option': controller_option[args.controller_option],
            'embedding_option': embedding_option[args.embedding_option],
            'root_option': root_option[args.root_option],

            'drop_body_nodes': args.drop_body_nodes,
            'use_sibling_relations': args.use_sibling_relations,
            'xml_assets_path': args.xml_assets_path,
            'policy_readout_mode': args.policy_readout_mode
        }
    alg_kwargs = args.__dict__.copy()
    alg_kwargs.pop("config", None)
    alg_kwargs.pop("task_name", None)
    alg_kwargs.pop("policy", None)
    alg_kwargs.pop("activation_fn", None)
    alg_kwargs.pop("gnn_for_values", None)
    alg_kwargs.pop("embedding_option", None)
    alg_kwargs.pop("controller_option", None)
    alg_kwargs.pop("root_option", None)
    alg_kwargs.pop("xml_assets_path", None)
    alg_kwargs.pop("alg", None)
    alg_kwargs.pop("net_arch", None)
    alg_kwargs.pop("experiment_name", None)
    alg_kwargs.pop("job_dir", None)
    alg_kwargs.pop("total_timesteps", None)
    alg_kwargs.pop("model_name", None)
    alg_kwargs.pop("n_envs", None)
    alg_kwargs.pop("drop_body_nodes", None)
    alg_kwargs.pop("use_sibling_relations", None)
    alg_kwargs.pop("experiment_name_suffix", None)
    alg_kwargs.pop("policy_readout_mode", None)

    model = alg_class(args.policy,
                      env,
                      verbose=1,
                      #   n_steps=args.n_steps,
                      policy_kwargs=policy_kwargs,
                      #   device=args.device,
                      #   tensorboard_log=args.tensorboard_log,
                      #   learning_rate=args.learning_rate,
                      #   batch_size=args.batch_size,
                      #   n_epochs=args.n_epochs,
                      **alg_kwargs)

    model.learn(total_timesteps=args.total_timesteps,
                callback=callbacks,
                tb_log_name=log_name)

    model.save(os.path.join(args.tensorboard_log +
                            "/" + log_name, args.model_name))


def dir_path(path):
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'),
                   default='configs/GNN_AntBulletEnv-v02.yaml')
    p.add_argument('--task_name', help='The name of the environment to use')
    p.add_argument('--xml_assets_path',
                   help="The path to the directory where the xml of the task's robot is defined",
                   type=Path,
                   default=Path(pybullet_data.getDataPath()) / "mjcf")

    p.add_argument('--alg', help='The algorithm to be used for training',
                   choices=["A2C", "PPO"])
    p.add_argument('--policy',
                   help='The type of model to use.',
                   choices=["GnnPolicy", "GnnPolicy_V0", "MlpPolicy"])
    p.add_argument("--total_timesteps",
                   help="The total number of samples (env steps) to train on",
                   type=int,
                   default=1000000)
    p.add_argument('--tensorboard_log',
                   help='the log location for tensorboard (if None, no logging)',
                   default="runs")
    p.add_argument('--n_steps',
                   help='The number of steps to run for each environment per update',
                   type=int,
                   default=1024)
    p.add_argument('--batch_size',
                   help='The number of steps to run for each environment per update',
                   type=int,
                   default=64)
    p.add_argument('--n_epochs',
                   help="For PPO: Number of epochs when optimizing the surrogate loss.",
                   type=int,
                   default=10)
    p.add_argument('--n_envs',
                   help="Number of environments to run in parallel to collect rollout. Each environment requires one CPU",
                   type=int,
                   default=2)

    p.add_argument('--seed', help='Random seed',
                   type=int,
                   default=1)
    p.add_argument('--device',
                   help='Device (cpu, cuda, ...) on which the code should be run.'
                        'Setting it to auto, the code will be run on the GPU if possible.',
                   default="auto")
    p.add_argument('--net_arch',
                   help='The specification of the policy and value networks',
                   type=json.loads)
    p.add_argument('--gnn_for_values',
                   type=bool,
                   help='whether or not to use the GNN for the value function',
                   default=False)
    p.add_argument('--policy_readout_mode',
                   help='what type of readout net to use.',
                   choices=["action_per_controller", "pooled",
                            "pooled_by_group", "flattened"],
                   default='flattened')
    p.add_argument('--activation_fn',
                   help='Activation function of the policy and value networks.',
                   choices=["Tanh", "ReLU"])

    p.add_argument('--controller_option',
                   help='Controller Option for mujoco parser',
                   choices=["shared", "unified", "seperate"],
                   default='shared')
    p.add_argument('--embedding_option',
                   help='Embedding Option for mujoco parser',
                   choices=["shared", "unified"],
                   default='shared')
    p.add_argument('--root_option',
                   help='Root Option for mujoco parser',
                   choices=["none", "body", "all"],
                   default='none')

    p.add_argument('--drop_body_nodes',
                   help='Whether or not to use body nodes or only the joints and root nodes. Option is passed to the mujoco parser',
                   type=bool,
                   default=False)
    p.add_argument('--use_sibling_relations',
                   help='',
                   type=bool,
                   default=False)
    p.add_argument('--learning_rate',
                   help='Learning rate value for the optimizers.',
                   type=float,
                   default=3.0e-4)
    p.add_argument('--job_dir', help='GCS location to export models')
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

    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list) and arg_dict[key] is not None:
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    args.learning_rate = float(args.learning_rate)
    if isinstance(args.xml_assets_path, str):
        args.xml_assets_path = Path(args.xml_assets_path)

    if args.experiment_name is None:
        policy_abbrv = args.policy.split("Policy")[0].upper()

        net_arch_desc = ""
        if isinstance(args.net_arch, dict):  # GNN net arch
            for net_arch_part in args.net_arch.keys():
                net_arch_desc += "_" + net_arch_part[:3]
                for i, (layer_class_name, layer_size) in enumerate(args.net_arch[net_arch_part]):
                    if isinstance(layer_size, list):
                        layer_size = "".join([str(i) for i in layer_size])
                    net_arch_desc += f"_{layer_size}"
        elif isinstance(args.net_arch, list):
            for net_arch_info in args.net_arch:
                if isinstance(net_arch_info, dict):
                    for net_arch_key in net_arch_info.keys():
                        net_arch_desc += "_" + net_arch_key + \
                                         "_".join([str(i)
                                                   for i in net_arch_info[net_arch_key]])
                else:
                    net_arch_desc += f"_{net_arch_info}"
        args.experiment_name = f"{policy_abbrv}_{args.alg}{net_arch_desc}_N{args.n_steps}_B{args.batch_size}_"
        args.experiment_name += f"lr{args.learning_rate:.0e}_"
        #args.experiment_name += f"GNNValue_{args.gnn_for_values:0d}_EmbOpt_{args.embedding_option}_"
        args.experiment_name += f"mode_{args.policy_readout_mode}_"
        args.experiment_name += f"Epochs_{args.n_epochs}_Nenvs_{args.n_envs}"

        if args.experiment_name_suffix is not None:
            args.experiment_name += f"_{args.experiment_name_suffix}"

    return args


if __name__ == '__main__':
    train(parse_arguments())
