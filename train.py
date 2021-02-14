import argparse
import copy
import json
import os

from datetime import datetime

import json
import pyaml
import torch
import yaml
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import MlpPolicy
from torch import nn

import pybullet_envs  # register pybullet envs from bullet3

from NerveNet.graph_util.mujoco_parser_settings import EmbeddingOption
from NerveNet.models.nerve_net_conv import NerveNetConv
from NerveNet.policies import register_policies
import NerveNet.gym_envs.pybullet.register_disability_envs

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from util import LoggingCallback

algorithms = dict(A2C=A2C, PPO=PPO)
activation_functions = dict(Tanh=nn.Tanh, ReLU=nn.ReLU)
embedding_option = dict(shared=EmbeddingOption.SHARED,
                        unified=EmbeddingOption.UNIFIED)


def train(args):
    cuda_availability = torch.cuda.is_available()
    print('\n*************************')
    print('`CUDA` available: {}'.format(cuda_availability))
    print('Device specified: {}'.format(args.device))
    print('*************************\n')

    # Create the environment
    env = gym.make(args.task_name)

    # define network architecture
    if args.policy == "GnnPolicy":
        net_arch = {
            "input": [
                (nn.Linear, 8)
            ],
            "propagate": [
                (NerveNetConv, 16),
                (NerveNetConv, 16),
                (NerveNetConv, 16),
            ],
            "policy": [
                (nn.Linear, 16),
            ],
            "value": [
                (nn.Linear, 16),
            ]
        }
    else:
        # for mlppolicy
        net_arch = [64, {"pi": [64], "vf": [64]}]

    # Prepare tensorboard logging
    log_name = '{}_{}_{}'.format(
        args.experiment_name, args.task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
    run_dir = args.tensorboard_log + "/" + log_name
    os.mkdir(run_dir)
    callbacks = []
    # callbacks.append(CheckpointCallback(
    #    save_freq=1000000, save_path=run_dir, name_prefix='rl_model'))
    callbacks.append(LoggingCallback(logpath=run_dir))
    with open(os.path.join(run_dir, 'net_arch.txt'), 'w') as fp:
        fp.write(str(net_arch))
    train_args = copy.copy(args)
    train_args.config = train_args.config.name
    pyaml.dump(train_args.__dict__, open(
        os.path.join(run_dir, 'train_arguments.yaml'), 'w'))

    # Create the model
    alg_class = algorithms[args.alg]
    alg_kwargs = dict()
    policy_kwargs = dict()
    policy_kwargs['net_arch'] = net_arch
    if args.activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_functions[args.activation_fn]
    # policy_kwargs['device'] = args.device if args.device is not None else get_device('auto')
    if args.policy == "GnnPolicy":
        policy_kwargs["mlp_extractor_kwargs"] = {
            "task_name": args.task_name,
            'device': args.device,
            'gnn_for_values': args.gnn_for_values,
            'embedding_option': embedding_option[args.embedding_option]
        }

    model = alg_class(args.policy,
                      env,
                      verbose=1,
                      n_steps=args.n_steps,
                      policy_kwargs=policy_kwargs,
                      device=args.device,
                      tensorboard_log=args.tensorboard_log,
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      n_epochs=args.n_epochs,
                      **alg_kwargs)

    model.learn(total_timesteps=args.total_timesteps,
                callback=callbacks,
                tb_log_name=log_name)

    model.save(os.path.join(args.tensorboard_log +
                            "/" + log_name, args.model_name))


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'),
                   default='configs/GNN_AntBulletEnv-v0.yaml')
    p.add_argument('--task_name', help='The name of the environment to use')
    p.add_argument(
        '--alg', help='The algorithm to be used for training', choices=["A2C", "PPO"])
    p.add_argument('--policy', help='The type of model to use.',
                   choices=["GnnPolicy", "MlpPolicy"])
    p.add_argument("--total_timesteps", help="The total number of samples (env steps) to train on", type=int,
                   default=1000000)
    p.add_argument('--tensorboard_log',
                   help='the log location for tensorboard (if None, no logging)', default="runs")
    p.add_argument('--n_steps', help='The number of steps to run for each environment per update', type=int,
                   default=1024)
    p.add_argument('--batch_size', help='The number of steps to run for each environment per update', type=int,
                   default=64)
    p.add_argument('--n_epochs', help="For PPO: Number of epochs when optimizing the surrogate loss.", type=int,
                   default=10)
    p.add_argument('--seed', help='Random seed', type=int, default=1)
    p.add_argument('--device', help='Device (cpu, cuda, ...) on which the code should be run.'
                                    'Setting it to auto, the code will be run on the GPU if possible.', default="auto")
    p.add_argument(
        '--net_arch', help='The specification of the policy and value networks', type=json.loads)
    p.add_argument('--gnn_for_values', type=bool, help='whether or not to use the GNN for the value function',
                   default=False)
    p.add_argument('--activation_fn', help='Activation function of the policy and value networks',
                   choices=["Tanh", "ReLU"])
    p.add_argument('--embedding_option', help='Embedding Option for mujoco parser',
                   choices=["shared", "unified"], default='shared')
    p.add_argument('--learning_rate',
                   help='Learning rate value for the optimizers.', type=float, default=3.0e-4)
    p.add_argument('--job_dir', help='GCS location to export models')
    p.add_argument('--experiment_name',
                   help='name to append to the tensorboard logs directory', default='')
    p.add_argument(
        '--model_name', help='The name of your saved model', default='model.zip')
    args = p.parse_args()
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    train(parse_arguments())
