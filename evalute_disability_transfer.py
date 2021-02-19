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
from NerveNet.models import nerve_net_conv
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
