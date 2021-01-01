from typing import Any, Dict, List, Optional, Tuple, Type, Union
from itertools import zip_longest

from pathlib import Path
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing


from stable_baselines3.common.utils import get_device
from graph_util.mujoco_parser import parse_mujoco_graph
from graph_util.observation_mapper import observations_to_node_attributes, relation_matrix_to_adjacency_matrix


class NerveNetGNN(nn.Module):
    """
    GNN from NerveNet paper:
        Wang, et al.
        "Nervenet: Learning structured policy with graph neural networks"
        6th International Conference on Learning Representations, ICLR 2018

    """

    def __init__(self,
                 feature_dim: int,
                 net_arch: List[Union[int, Dict[str, List[int]]]],
                 activation_fn: Type[nn.Module],
                 device: Union[torch.device, str] = "auto",
                 task_name: str = None,
                 xml_name: str = None,
                 xml_assets_path: Path = None):
        '''
        TODO add documentation
        '''
        super(NerveNetGNN, self).__init__()

        self.task_name = task_name
        self.xml_name = xml_name
        self.xml_assets_path = xml_assets_path

        self.info = parse_mujoco_graph(task_name=self.task_name,
                                       xml_name=self.xml_name,
                                       xml_assets_path=self.xml_assets_path)

        # Notes on edge attributes:
        # using one hot encoding leads to num_edge_features != 1
        # officially this is supported for graph data types.
        # However, depending on the type of GNN used, the edge attributes are
        # interpreted not as attributes but as weights.
        # Hence, they can't be of arbitrary shape (must be [num_edges, 1]) and
        # should be somewhat meaningfully be interpretable as weight factors.
        # This is not the case for attributes produced by the following function!
        # Ergo, we should not use them!
        self.edge_index, self.edge_attr = relation_matrix_to_adjacency_matrix(
            self.info["relation_matrix"],
            one_hot_attributes=True,
            self_loop=True
        )

        self.device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        # Layer sizes of the network that only belongs to the policy network
        policy_only_layers = []
        # Layer sizes of the network that only belongs to the value network
        value_only_layers = []
        last_layer_dim_shared = self.info["num_node_features"]

        # from here on we build the network

        # Iterate through the shared layers and build the shared parts of the network
        # only the shared network will have GCN convolutions
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(GCNConv(last_layer_dim_shared,
                                          layer_size,
                                          # we already added self_loops ourselves
                                          add_self_loops=False))

                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(
                    layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(
                        layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(
                        layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # in the shared network we use GCN convolutions, which means last_layer_dim_shared is the number of
        # dimensions we have for every single node
        last_layer_dim_pi = self.info["num_nodes"] * last_layer_dim_shared
        last_layer_dim_vf = self.info["num_nodes"] * last_layer_dim_shared

        # Build the non-shared part of the network
        # these will be plain old MLPs
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = shared_net
        self.policy_net = nn.Sequential(*policy_net).to(self.device)
        self.value_net = nn.Sequential(*value_net).to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
            return: 
                latent_policy, latent_value of the specified network.
                If all layers are shared, then ``latent_policy == latent_value``
         """
        shared_latent = observations_to_node_attributes(observations,
                                                        self.info["obs_input_mapping"],
                                                        self.info["static_input_mapping"],
                                                        self.info["num_nodes"],
                                                        self.info["num_node_features"]
                                                        )
        for layer in self.shared_net:
            if isinstance(layer, MessagePassing):
                shared_latent = layer(shared_latent, self.edge_index)
            else:
                shared_latent = layer(shared_latent)

        shared_latent = shared_latent.flatten()[None, :]
        latent_pi, latent_vf = self.policy_net(
            shared_latent), self.value_net(shared_latent)
        return latent_pi, latent_vf
