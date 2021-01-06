import argparse
import json
import os

from datetime import datetime

import torch
import yaml
from torch import nn

import pybullet_envs  # register pybullet envs from bullet3
from NerveNet.policies import register_policies

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

algorithms = dict(A2C=A2C,
                  PPO=PPO)

activation_functions = dict(Tanh=nn.Tanh,
                            ReLU=nn.ReLU)


def train(args):
    cuda_availability = torch.cuda.is_available()
    print('\n*************************')
    print('`CUDA` available: {}'.format(cuda_availability))
    print('Device specified: {}'.format(args.device))
    print('*************************\n')

    # Create the environment
    print(args)
    env = gym.make(args.task_name)

    # Prepare tensorboard logging
    log_name = '{}_{}'.format(
        args.task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=args.tensorboard_log + "/" + log_name,
                                             name_prefix='rl_model')
    # Create the model
    alg_class = algorithms[args.alg]
    alg_kwargs = dict()
    policy_kwargs = dict()
    if args.activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_functions[args.activation_fn]
    if args.policy == "GnnPolicy":
        policy_kwargs["mlp_extractor_kwargs"] = {
            "task_name": args.task_name}

    model = alg_class(args.policy,
                      env,
                      verbose=1,
                      n_steps=args.n_steps,
                      policy_kwargs=policy_kwargs,
                      device=args.device,
                      tensorboard_log=args.tensorboard_log,
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      **alg_kwargs)

    # Train / Test the model
    model.learn(total_timesteps=args.total_timesteps,
                callback=checkpoint_callback,
                tb_log_name=log_name)

    model.save(os.path.join(args.tensorboard_log + "/" + log_name, args.model_name))


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config.yaml')
    p.add_argument('--task_name', help='The name of the environment to use')
    p.add_argument('--alg', help='The algorithm to be used for training', choices=["A2C", "PPO"])
    p.add_argument('--policy', help='The type of model to use.', choices=["GnnPolicy", "MlpPolicy"])
    p.add_argument("--total_timesteps", help="The total number of samples (env steps) to train on", type=int,
                   default=10000)
    p.add_argument('--tensorboard_log', help='the log location for tensorboard (if None, no logging)', default="runs")
    p.add_argument('--n_steps', help='The number of steps to run for each environment per update', type=int,
                   default=1024)
    p.add_argument('--batch_size', help='The number of steps to run for each environment per update', type=int,
                   default=64)
    p.add_argument('--n_epochs', help="For PPO: Number of epochs when optimizing the surrogate loss.", type=int)
    p.add_argument('--seed', help='Random seed', type=int, default=1)
    p.add_argument('--device', help='Device (cpu, cuda, ...) on which the code should be run.'
                                    'Setting it to auto, the code will be run on the GPU if possible.', default="auto")
    p.add_argument('--net_arch', help='The specification of the policy and value networks', type=json.loads)
    p.add_argument('--activation_fn', help='Activation function of the policy and value networks',
                   choices=["Tanh", "ReLU"])
    p.add_argument('--learning_rate', help='Learning rate value for the optimizers.', type=float, default=3.0e-4)
    p.add_argument('--job_dir', help='GCS location to export models')
    p.add_argument('--model_name', help='The name of your saved model', default='model.pth')
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
