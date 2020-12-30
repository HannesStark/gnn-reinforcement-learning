from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pathlib import Path
import gym
import numpy as np
import torch
import torch.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, \
    create_mlp
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from graph_util.mujoco_parser import parse_mujoco_graph
from graph_util.observation_mapper import observations_to_node_attributes


class NerveNetGNN(BaseFeaturesExtractor):
    """
    GNN from NerveNet paper:
        Wang, et al.
        "Nervenet: Learning structured policy with graph neural networks"
        6th International Conference on Learning Representations, ICLR 2018

    :param observation_space:
    :param env:
    """

    def __init__(self,
                 observation_space: gym.Space,
                 task_name: str = None,
                 xml_name: str = None,
                 xml_assets_path: Path = None):
        '''

        :param observation_space:
        :param env:
        :param agent_structure: Dictionary of the agent defined by an XML extracted by parser_ours.py
        :param task_name:
        :param xml_name:
        :param xml_assets_path:
        '''
        super(NerveNetGNN, self).__init__(observation_space,
                                          get_flattened_obs_dim(observation_space))
        # TODO: either require number of features to be given as argument or extract them from env

        self.task_name = task_name
        self.xml_name = xml_name
        self.xml_assets_path = xml_assets_path

        self.info = parse_mujoco_graph(task_name=self.task_name,
                                       xml_name=self.xml_name,
                                       xml_assets_path=self.xml_assets_path)
        # self.adj_matrix =
        self.conv1 = GCNConv(1, 1, node_dim=1)
        self.conv2 = GCNConv(1, 1, node_dim=1)

        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations[..., None]  # [batchsize,num_nodes, num_node_features]
        x = self.conv1(x, self.relation_matrix)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, self.relation_matrix)
        return self.flatten(x)
