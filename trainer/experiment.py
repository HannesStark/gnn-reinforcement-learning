# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
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


def run(args):
    """Create the environment, train, evaluate, and export the model 

    Args:
    args: experiment parameters.
    """
    cuda_availability = torch.cuda.is_available()
    device = args.device
    print('\n*************************')
    print('`CUDA` available: {}'.format(cuda_availability))
    print('Device specified: {}'.format(device))
    print('*************************\n')

    # Create the environment
    env = gym.make(args.task_name)

    # Prepare tensorboard logging
    log_name = '{}_{}'.format(
        task_name, datetime.now().strftime('%d-%m_%H-%M-%S'))
    checkpoint_callback = CheckpointCallback(save_freq=1000,
                                             save_path=args.tensorboard_log + log_name,
                                             name_prefix='rl_model')
    # Create the model
    alg_class = algorithms[args.alg]
    alg_kwargs = dict()
    update_not_none(alg_kwargs, args, "n_steps")
    update_not_none(alg_kwargs, args, "n_epochs")
    update_not_none(alg_kwargs, args, "seed")

    policy_kwargs = dict()
    update_not_none(alg_kwargs, args, "net_arch")
    policy_kwargs["activation_fn"] = activation_functions[args.activation_fn]
    if args.policy == "GnnPolicy":
        policy_kwargs["mlp_extractor_kwargs"] = {"task_name": args.task_name}

    model = alg_class(args.policy,
                      env,
                      verbose=1,
                      policy_kwargs=policy_kwargs,
                      device=device,
                      tensorboard_log=args.tensorboard_log,
                      **alg_kwargs)

    # Train / Test the model
    model.learn(total_timesteps=args.total_timesteps,
                callback=checkpoint_callback,
                tb_log_name=log_name)

    model.save(args.model_name)

    # TODO Upload the model to GCS


def update_not_none(arg_dict: dict, args, argument: str):
    """
    Inplace update of argument dictionary if argument is not none
    """
    if args[argument] is not None:
        alg_kwargs[argument] = args[argument]
