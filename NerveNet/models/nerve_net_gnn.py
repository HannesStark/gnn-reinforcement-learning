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
from NerveNet.graph_util.mujoco_parser import parse_mujoco_graph
from NerveNet.graph_util.mujoco_parser_settings import EmbeddingOption, RootRelationOption
from NerveNet.graph_util.observation_mapper import get_update_masks, observations_to_node_attributes, \
    relation_matrix_to_adjacency_matrix, get_static_node_attributes
from NerveNet.models.nerve_net_conv import NerveNetConv, NerveNetConv_v1, NerveNetConvGRU, NerveNetConvGAT


class NerveNetGNN(nn.Module):
    """
    GNN from NerveNet paper:
        Wang, et al.
        "Nervenet: Learning structured policy with graph neural networks"
        6th International Conference on Learning Representations, ICLR 2018

    """

    def __init__(self,
                 net_arch: Dict[str, List[Tuple[nn.Module, int]]],
                 activation_fn: Type[nn.Module],
                 gnn_for_values=False,
                 action_per_controller=True,
                 use_sibling_relations: bool = False,
                 drop_body_nodes: bool = True,
                 embedding_option=EmbeddingOption.SHARED,
                 device: Union[torch.device, str] = "auto",
                 task_name: str = None,
                 xml_name: str = None,
                 xml_assets_path: Path = None):
        '''
        TODO add documentation

        Parameters:
            net_arch:
                Specifies the network architecture. The network consists of four parts:
                First we have two parts that make out the shared network. This takes as
                input the observations mapped to the node embedding space.
                The mapping is done based on the group a node belongs to (hips, feet, ankles, etc.).
                Because this mapping results in differently sized node features (e.g. ankles
                may have more node features than feet) the first part of the shared network
                is called the input model, which produces a fixed-size node embedding vector
                for all nodes regardless of their group.
                The second part of the shared network is a GNN which is called the propagation model.
                It takes the fixed-size embedding vectors and the adjacency matrix and outputs the new
                node embeddings.
                Afterwards we have two seperate networks, the value model and policy model.
                Both take the new node embeddings and output a latent representation for the policy mean
                or the value scalar

                The network architecture is provided as a dictionary of lists with four keys
                corresponding to the four parts of the network as described above.
                Each list is a list of tuples of type (nn.Module, int) where the first element
                is the layer class that should be used and the second element is the output
                size of this layer.

                For exmaple:
                net_arch = {
                    "input": [
                        (nn.Linear, 8)
                    ],
                    "propagate": [
                        (GCNConv, 12),
                        (nn.Linear, 16),
                        (GCNConv, 12)
                    ],
                    "policy": [
                        (nn.Linear, 16)
                    ],
                    "value": [
                        (nn.Linear, 16)
                    ]
                }
        '''
        super(NerveNetGNN, self).__init__()

        self.task_name = task_name
        self.xml_name = xml_name
        self.xml_assets_path = xml_assets_path
        self.device = get_device(device)
        self.gnn_for_values = gnn_for_values
        self.action_per_controller = action_per_controller

        self.info = parse_mujoco_graph(task_name=self.task_name,
                                       xml_name=self.xml_name,
                                       use_sibling_relations=use_sibling_relations,
                                       drop_body_nodes=drop_body_nodes,
                                       root_relation_option=RootRelationOption.NONE,
                                       xml_assets_path=self.xml_assets_path,
                                       embedding_option=embedding_option)

        self.action_node_indices = self.info['output_list']

        self.info["static_input_mapping"] = {}
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
            self_loop=True
        )

        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.static_node_attr, self.static_node_attr_mask = get_static_node_attributes(
            self.info["static_input_mapping"],
            self.info["num_nodes"])
        self.update_masks, self.observation_mask = get_update_masks(self.info["obs_input_mapping"],
                                                                    self.static_node_attr_mask,
                                                                    self.static_node_attr.shape,
                                                                    self.info["input_type_dict"])

        self.shared_input_nets = nn.ModuleDict()
        gnn_policy = nn.ModuleList()
        gnn_values = nn.ModuleList()
        value_net = nn.ModuleList()
        # Layer sizes of the network that only belongs to the policy network
        policy_only_layers = []
        # Layer sizes of the network that only belongs to the value network
        value_only_layers = []

        assert "input" in net_arch, "An input model must be specified in the net_arch attribute"
        assert "propagate" in net_arch, "A propagation model must be specified in the net_arch attribute"
        assert "policy" in net_arch, "A policy model must be specified in the net_arch attribute"
        assert "value" in net_arch, "A value model must be specified in the net_arch attribute"

        # from here on we build the network
        # first we build the input model, where each group of nodes gets
        # its own instance of the input model
        for group_name, (_, attribute_mask) in self.update_masks.items():
            shared_input_layers = []
            last_layer_dim_input = len(attribute_mask)
            if last_layer_dim_input > 0:
                for layer_class, layer_size in net_arch["input"]:
                    shared_input_layers.append(layer_class(
                        last_layer_dim_input, layer_size))
                    shared_input_layers.append(activation_fn())
                    last_layer_dim_input = layer_size
            else:
                shared_input_layers.append(nn.Identity())
                last_layer_dim_input = net_arch["input"][-1][1]

            self.shared_input_nets[group_name] = nn.Sequential(
                *shared_input_layers).to(self.device)

        # max_static_feature_dim = self.static_node_attr.shape[1]
        # max_obs_feature_dim = max(
        #     [len(l) for l in self.info["obs_input_mapping"].values()])
        # last_layer_dim_input = max_obs_feature_dim + max_static_feature_dim

        self.last_layer_dim_input = last_layer_dim_input
        last_layer_dim_shared = last_layer_dim_input

        # Iterate through the shared layers and build the shared parts of the network
        # only the shared network may have GCN convolutions
        for layer_class, layer_size in net_arch["propagate"]:
            # TODO: give layer a meaningful name
            if issubclass(layer_class, NerveNetConv):
                gnn_values.append(layer_class(last_layer_dim_shared,
                                              layer_size,
                                              self.update_masks, device=device).to(self.device))
                gnn_policy.append(layer_class(last_layer_dim_shared,
                                              layer_size,
                                              self.update_masks, device=device).to(self.device))
            elif layer_class == NerveNetConvGRU:
                gnn_values.append(layer_class(*layer_size,
                                              self.update_masks).to(self.device))
                gnn_policy.append(layer_class(*layer_size,
                                              self.update_masks).to(self.device))
            elif layer_class == NerveNetConvGAT:
                gnn_values.append(layer_class(last_layer_dim_shared,
                                              *layer_size[:],
                                              # we already added self_loops ourselves
                                              add_self_loops=False).to(self.device))
                gnn_policy.append(layer_class(last_layer_dim_shared,
                                              *layer_size[:],
                                              # we already added self_loops ourselves
                                              add_self_loops=False).to(self.device))

            elif issubclass(layer_class, GCNConv):
                # for GCN Conv we need an additional parameter for the constructor
                gnn_values.append(layer_class(last_layer_dim_shared,
                                              layer_size,
                                              # we already added self_loops ourselves
                                              add_self_loops=False).to(self.device))
                gnn_policy.append(layer_class(last_layer_dim_shared,
                                              layer_size,
                                              # we already added self_loops ourselves
                                              add_self_loops=False).to(self.device))
            else:
                gnn_values.append(layer_class(last_layer_dim_shared,
                                              layer_size).to(self.device))
                gnn_policy.append(layer_class(last_layer_dim_shared,
                                              layer_size).to(self.device))
            gnn_policy.append(activation_fn())
            gnn_values.append(activation_fn())
            if layer_class == NerveNetConv_v1:
                # NerveNetConv_v1 uses a form of skip connection, the layer output needs to be calculated accordingly
                last_layer_dim_shared = last_layer_dim_shared + layer_size
            elif layer_class == NerveNetConvGRU:
                last_layer_dim_shared = layer_size[0]
            elif layer_class == NerveNetConvGAT:
                last_layer_dim_shared = layer_size[0] * layer_size[1]
            else:
                last_layer_dim_shared = layer_size

        # Build the non-shared part of the network

        if self.gnn_for_values:
            last_layer_dim_vf = self.info["num_nodes"] * last_layer_dim_shared
            vf_net_dim = last_layer_dim_shared
        else:
            last_layer_dim_vf = self.info["num_nodes"] * \
                self.last_layer_dim_input
            vf_net_dim = last_layer_dim_vf  # self.last_layer_dim_input

        if not self.action_per_controller:
            # use the features of all nodes to generate an action, not just the features of the controller node
            # in the shared network we use GCN convolutions,
            # which means last_layer_dim_shared is the number of
            # dimensions we have for every single node
            last_layer_dim_shared = self.info["num_nodes"] * \
                last_layer_dim_shared

        self.policy_nets = dict()
        for out_group_name, out_node_idx in self.info["output_type_dict"].items():
            policy_net = nn.ModuleList()
            policy_net_dim = last_layer_dim_shared
            for layer_class, layer_size in net_arch["policy"]:
                policy_net.append(layer_class(
                    policy_net_dim, layer_size).to(self.device))
                policy_net.append(activation_fn().to(self.device))
                policy_net_dim = layer_size
            # add mandatory linear layer that returns a scalar for each node
            if self.action_per_controller:
                policy_net.append(nn.Linear(policy_net_dim, 1).to(self.device))
            else:
                policy_net.append(
                    nn.Linear(policy_net_dim, len(out_node_idx)).to(self.device))

            self.policy_nets[out_group_name] = nn.Sequential(
                *policy_net).to(self.device)

        for layer_class, layer_size in net_arch["value"]:
            value_net.append(layer_class(
                vf_net_dim, layer_size).to(self.device))
            value_net.append(activation_fn().to(self.device))
            vf_net_dim = layer_size
        # add mandatory linear layer that returns a scalar for the pooled embeddings
        value_net.append(nn.Linear(vf_net_dim, 1).to(self.device))

        # Save dim, used to create the distributions
        self.latent_dim_pi = policy_net_dim
        self.latent_dim_vf = vf_net_dim

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.gnn_policy = gnn_policy
        self.gnn_values = gnn_values
        self.flatten = nn.Flatten()
        self.value_net = nn.Sequential(*value_net).to(self.device)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            return:
                latent_policy, latent_value of the specified network.
                If all layers are shared, then ``latent_policy == latent_value``
         """
        # "sparse" embedding matrix
        sp_embedding = observations_to_node_attributes(observations,
                                                       self.info["obs_input_mapping"],
                                                       self.observation_mask,
                                                       self.update_masks,
                                                       self.static_node_attr,
                                                       self.static_node_attr_mask,
                                                       self.info["num_nodes"]
                                                       ).to(self.device)
        # dense embedding matrix
        embedding = torch.zeros(
            (*sp_embedding.shape[:-1], self.last_layer_dim_input)).to(self.device)

        for group_name, (node_mask, attribute_mask) in self.update_masks.items():
            if len(attribute_mask) > 0:
                embedding[:, node_mask, :] = self.shared_input_nets[group_name](
                    sp_embedding[:, node_mask][:, :, attribute_mask])

        # embedding = sp_embedding
        # [batch_size, number_nodes, features_dim]
        pre_message_passing = embedding

        policy_embedding = embedding
        for layer in self.gnn_policy:
            if isinstance(layer, NerveNetConvGAT):
                # GATs don't support batched input...
                # for i in range(embedding.shape[0]):
                #     policy_embedding[[i], :, :] = layer(
                #         policy_embedding[i], self.edge_index).to(self.device)
                policy_embedding = layer(
                    policy_embedding, self.edge_index).to(self.device)

            elif isinstance(layer, MessagePassing):
                policy_embedding = layer(policy_embedding, self.edge_index,
                                         self.update_masks).to(self.device)  # [batch_size, number_nodes, features_dim]
            else:
                policy_embedding = layer(policy_embedding).to(
                    self.device)  # [batch_size, number_nodes, features_dim]

        value_embedding = embedding
        for layer in self.gnn_values:
            if isinstance(layer, NerveNetConvGAT):
                # GATs don't support batched input...
                # for i in range(embedding.shape[0]):
                #     value_embedding[[i], :, :] = layer(
                #         value_embedding[i], self.edge_index).to(self.device)
                value_embedding = layer(
                    value_embedding, self.edge_index).to(self.device)

            elif isinstance(layer, MessagePassing):
                value_embedding = layer(value_embedding, self.edge_index,
                                        self.update_masks).to(self.device)  # [batch_size, number_nodes, features_dim]
            else:
                # [batch_size, number_nodes, features_dim]
                value_embedding = layer(value_embedding).to(self.device)

        if self.gnn_for_values:
            pooled_embedding = torch.mean(value_embedding, dim=1)
        else:
            pooled_embedding = self.flatten(pre_message_passing)

        latent_vf = self.value_net(pooled_embedding)
        # latent_vf = self.debug(self.flatten(pre_message_passing))

        action_nodes_embedding = policy_embedding[:, self.action_node_indices,
                                                  :]  # [batchsize, number_action_nodes, features_dim]
        action_nodes_embedding_flat = action_nodes_embedding.view(-1, action_nodes_embedding.shape[
            -1])  # [batchsize * number_action_nodes, features_dim]

        # for debugging
        # flat_embedding = self.flatten(embedding)  # for debug network
        # latent_pi = self.debug(flat_embedding)  # [batch_size * number_nodes, features_dim]
        # [batch_size * number_nodes, 1]

        latent_pis = torch.zeros(
            *observations.shape[:-1], len(self.action_node_indices))
        for out_group_name, out_node_idx in self.info["output_type_dict"].items():
            policy_net = self.policy_nets[out_group_name]
            if self.action_per_controller:
                for i in out_node_idx:
                    latent_pis[:, self.action_node_indices.index(i)] = policy_net(
                        action_nodes_embedding[:, self.action_node_indices.index(i), :]).view(*observations.shape[:-1])
            else:
                out_idx = [self.action_node_indices.index(
                    i) for i in out_node_idx]
                # input to policy_net is [batch_size, num_nodes * num_features ]
                latent_pis[:, out_idx] = policy_net(
                    policy_embedding.view(observations.shape[0], -1))

        # latent_pis = self.policy_net(action_nodes_embedding_flat) # [batch_size, number_nodes]
        # latent_pis = latent_pis.view(-1, action_nodes_embedding.shape[1])
        return latent_pis, latent_vf


# This is the old version of our NerveNetGNN from the intermediate presentation
class NerveNetGNN_V0(nn.Module):
    """
    GNN from NerveNet paper:
        Wang, et al.
        "Nervenet: Learning structured policy with graph neural networks"
        6th International Conference on Learning Representations, ICLR 2018
    """

    def __init__(self,
                 net_arch: Dict[str, List[Tuple[nn.Module, int]]],
                 activation_fn: Type[nn.Module],
                 gnn_for_values=False,
                 use_sibling_relations: bool = False,
                 drop_body_nodes: bool = True,
                 embedding_option=EmbeddingOption.SHARED,
                 device: Union[torch.device, str] = "auto",
                 task_name: str = None,
                 xml_name: str = None,
                 xml_assets_path: Path = None):
        '''
        TODO add documentation
        Parameters:
            net_arch:
                Specifies the network architecture. The network consists of four parts:
                First we have two parts that make out the shared network. This takes as
                input the observations mapped to the node embedding space.
                The mapping is done based on the group a node belongs to (hips, feet, ankles, etc.).
                Because this mapping results in differently sized node features (e.g. ankles
                may have more node features than feet) the first part of the shared network
                is called the input model, which produces a fixed-size node embedding vector
                for all nodes regardless of their group.
                The second part of the shared network is a GNN which is called the propagation model.
                It takes the fixed-size embedding vectors and the adjacency matrix and outputs the new
                node embeddings.
                Afterwards we have two seperate networks, the value model and policy model.
                Both take the new node embeddings and output a latent representation for the policy mean
                or the value scalar
                The network architecture is provided as a dictionary of lists with four keys
                corresponding to the four parts of the network as described above.
                Each list is a list of tuples of type (nn.Module, int) where the first element
                is the layer class that should be used and the second element is the output
                size of this layer.
                For exmaple:
                net_arch = {
                    "input": [
                        (nn.Linear, 8)
                    ],
                    "propagate": [
                        (GCNConv, 12),
                        (nn.Linear, 16),
                        (GCNConv, 12)
                    ],
                    "policy": [
                        (nn.Linear, 16)
                    ],
                    "value": [
                        (nn.Linear, 16)
                    ]
                }
        '''
        super(NerveNetGNN_V0, self).__init__()

        self.task_name = task_name
        self.xml_name = xml_name
        self.xml_assets_path = xml_assets_path
        self.device = get_device(device)
        self.gnn_for_values = gnn_for_values

        self.info = parse_mujoco_graph(task_name=self.task_name,
                                       xml_name=self.xml_name,
                                       xml_assets_path=self.xml_assets_path,
                                       embedding_option=embedding_option)
        self.info["static_input_mapping"] = {}
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
            self_loop=True
        )
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.static_node_attr, self.static_node_attr_mask = get_static_node_attributes(self.info["static_input_mapping"],
                                                                                       self.info["num_nodes"])
        self.update_masks, self.observation_mask = get_update_masks(self.info["obs_input_mapping"],
                                                                    self.static_node_attr_mask,
                                                                    self.static_node_attr.shape,
                                                                    self.info["input_type_dict"])

        self.shared_input_nets = {}
        shared_net, policy_net, value_net = [], [], []
        # Layer sizes of the network that only belongs to the policy network
        policy_only_layers = []
        # Layer sizes of the network that only belongs to the value network
        value_only_layers = []

        assert "input" in net_arch, "An input model must be specified in the net_arch attribute"
        assert "propagate" in net_arch, "A propagation model must be specified in the net_arch attribute"
        assert "policy" in net_arch, "A policy model must be specified in the net_arch attribute"
        assert "value" in net_arch, "A value model must be specified in the net_arch attribute"

        # from here on we build the network
        # first we build the input model, where each group of nodes gets
        # its own instance of the input model
        for group_name, (_, attribute_mask) in self.update_masks.items():
            shared_input_layers = []
            last_layer_dim_input = len(attribute_mask)
            if last_layer_dim_input > 0:
                for layer_class, layer_size in net_arch["input"]:
                    shared_input_layers.append(layer_class(
                        last_layer_dim_input, layer_size))
                    shared_input_layers.append(activation_fn())
                    last_layer_dim_input = layer_size
            else:
                shared_input_layers.append(nn.Identity())
                last_layer_dim_input = net_arch["input"][-1][1]

            self.shared_input_nets[group_name] = nn.Sequential(
                *shared_input_layers).to(self.device)

        # max_static_feature_dim = self.static_node_attr.shape[1]
        # max_obs_feature_dim = max(
        #     [len(l) for l in self.info["obs_input_mapping"].values()])
        # last_layer_dim_input = max_obs_feature_dim + max_static_feature_dim

        self.last_layer_dim_input = last_layer_dim_input
        last_layer_dim_shared = last_layer_dim_input

        # Iterate through the shared layers and build the shared parts of the network
        # only the shared network may have GCN convolutions
        for layer_class, layer_size in net_arch["propagate"]:
            # TODO: give layer a meaningful name
            if layer_class == GCNConv:
                # for GCN Conv we need an additional parameter for the constructor
                shared_net.append(layer_class(last_layer_dim_shared,
                                              layer_size,
                                              # we already added self_loops ourselves
                                              add_self_loops=False).to(self.device))
            elif layer_class == NerveNetConv:
                shared_net.append(layer_class(last_layer_dim_shared,
                                              layer_size,
                                              self.update_masks, device=device).to(self.device))
            else:
                shared_net.append(layer_class(last_layer_dim_shared,
                                              layer_size).to(self.device))
            shared_net.append(activation_fn())
            last_layer_dim_shared = layer_size

        # Build the non-shared part of the network

        # in the shared network we use GCN convolutions,
        # which means last_layer_dim_shared is the number of
        # dimensions we have for every single node
        last_layer_dim_pi = self.info["num_nodes"] * last_layer_dim_shared
        if self.gnn_for_values:
            last_layer_dim_vf = self.info["num_nodes"] * last_layer_dim_shared
        else:
            last_layer_dim_vf = self.info["num_nodes"] * \
                self.last_layer_dim_input

        for layer_class, layer_size in net_arch["policy"]:
            policy_net.append(layer_class(
                last_layer_dim_pi, layer_size).to(self.device))
            policy_net.append(activation_fn().to(self.device))
            last_layer_dim_pi = layer_size
        policy_net.append(nn.Linear(last_layer_dim_pi, len(
            self.info["output_list"])).to(self.device))

        for layer_class, layer_size in net_arch["value"]:
            value_net.append(layer_class(
                last_layer_dim_vf, layer_size).to(self.device))
            value_net.append(activation_fn().to(self.device))
            last_layer_dim_vf = layer_size
        value_net.append(nn.Linear(last_layer_dim_vf, 1).to(self.device))

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = shared_net
        self.flatten = nn.Flatten()
        self.policy_net = nn.Sequential(*policy_net).to(self.device)
        self.value_net = nn.Sequential(*value_net).to(self.device)
        self.debug = nn.Sequential(
            nn.Linear(self.last_layer_dim_input, 64),
            activation_fn(),
            nn.Linear(64, 64),
            activation_fn()
        )

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            return:
                latent_policy, latent_value of the specified network.
                If all layers are shared, then ``latent_policy == latent_value``
         """
        # "sparse" embedding matrix
        sp_embedding = observations_to_node_attributes(observations,
                                                       self.info["obs_input_mapping"],
                                                       self.observation_mask,
                                                       self.update_masks,
                                                       self.static_node_attr,
                                                       self.static_node_attr_mask,
                                                       self.info["num_nodes"]
                                                       ).to(self.device)

        # dense embedding matrix
        embedding = torch.zeros(
            (*sp_embedding.shape[:-1], self.last_layer_dim_input)).to(self.device)

        for group_name, (node_mask, attribute_mask) in self.update_masks.items():
            if len(attribute_mask) > 0:
                embedding[:, node_mask, :] = self.shared_input_nets[group_name](
                    sp_embedding[:, node_mask][:, :, attribute_mask])

        # embedding = sp_embedding

        pre_message_passing = self.flatten(embedding).to(self.device)

        for layer in self.shared_net:
            if isinstance(layer, MessagePassing):
                embedding = layer(embedding, self.edge_index,
                                  self.update_masks).to(self.device)
            else:
                embedding = layer(embedding).to(self.device)

        embedding = self.flatten(embedding).to(self.device)

        if self.gnn_for_values:
            latent_vf = self.value_net(embedding)
        else:
            latent_vf = self.value_net(pre_message_passing)

        latent_pi = self.policy_net(embedding)
        return latent_pi, latent_vf
