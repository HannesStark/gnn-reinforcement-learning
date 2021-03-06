import itertools
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import numpy as np
from scipy import sparse
from torch_geometric import utils


def flatten(t): return [item for sublist in t for item in sublist]


def relation_matrix_to_adjacency_matrix(relation_matrix: List[List],
                                        self_loop: bool = True):

    edge_matrix = sparse.csr_matrix(relation_matrix, dtype=np.long)

    if self_loop:
        edge_matrix += sparse.csr_matrix(np.diag(
            np.ones(edge_matrix.shape[0], dtype=np.long)))

    return utils.from_scipy_sparse_matrix(edge_matrix)


def get_static_node_attributes(static_input_mapping: dict,
                               num_nodes: int):
    if static_input_mapping == {}:
        return np.zeros((num_nodes, 0), dtype=np.float32), [np.array(mask, np.int64)
                                                            for mask in np.empty((num_nodes, 0)).tolist()]

    input_attributes = [list(zip(v.keys(), [len(l) for l in v.values()]))
                        for k, v in static_input_mapping.items()]
    input_attributes = list(set(flatten(input_attributes)))
    feature_dim = sum(list(zip(*input_attributes))[1])

    static_attributes = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    static_attribute_masks = np.empty((num_nodes, 0)).tolist()
    for node_id, node_attr in static_input_mapping.items():
        attr_offset_start = 0
        for attr_name, attr_dim_size in input_attributes:
            attr_offset_end = (attr_offset_start+attr_dim_size)
            if attr_name in node_attr.keys():
                static_attributes[node_id,
                                  attr_offset_start:attr_offset_end] = node_attr[attr_name]
                static_attribute_masks[node_id] += list(
                    range(attr_offset_start, attr_offset_end))
            attr_offset_start = attr_offset_end

    static_attribute_masks = [np.array(mask, np.int64)
                              for mask in static_attribute_masks]
    return static_attributes, static_attribute_masks


def observations_to_node_attributes(observations: torch.Tensor,
                                    obs_input_mapping: dict,
                                    observation_mask,
                                    update_masks,
                                    static_node_attr,
                                    static_node_attr_mask,
                                    num_nodes):

    batch_size, observation_sample_size = observations.shape
    # check that there is a mapping for every element of the observation vector
    assert max(itertools.chain.from_iterable(list(obs_input_mapping.values()))) + 1 == observation_sample_size

    max_static_feature_dim = static_node_attr.shape[1]
    max_obs_feature_dim = max([len(l) for l in obs_input_mapping.values()])
    num_node_features = max_obs_feature_dim + max_static_feature_dim
    with torch.no_grad():
        attributes = torch.zeros((batch_size,
                                  num_nodes,
                                  num_node_features), dtype=torch.float32, device=observations.device)
        if max_static_feature_dim != 0:
            attributes[:, :, -max_static_feature_dim:] = torch.from_numpy(
                static_node_attr).to(observations.device)

        for group_name, (node_mask, attr_mask) in update_masks.items():
            obs_attr_mask = attr_mask[attr_mask < max_obs_feature_dim]
            if len(obs_attr_mask) > 0:
                obs_mask_size = max(obs_attr_mask) + 1
                attributes[:, node_mask, 0:obs_mask_size] = observations[:, flatten(
                    observation_mask[group_name])].reshape(batch_size, len(node_mask), -1)
    return attributes


def get_update_masks(obs_input_mapping: dict,
                     static_node_attr_masks: List,
                     static_node_attr_shape,
                     input_type_dict: dict):
    """
        returns:
            update_mask:
                A dictionary containing groups of nodes that should share the
                same update function instance
    """

    max_obs_feature_dim = max([len(l) for l in obs_input_mapping.values()])
    update_mask = {}
    obs_mask = {}
    for group_name, group_nodes in input_type_dict.items():
        # assumes every node of a group has the same input sizes
        in_size = len(obs_input_mapping[group_nodes[0]])
        attribute_mask = np.concatenate([
            np.array(range(in_size)),
            (static_node_attr_masks[group_nodes[0]] + max_obs_feature_dim)])
        update_mask[group_name] = (group_nodes, attribute_mask)
        obs_mask[group_name] = [obs_input_mapping[node]
                                for node in group_nodes]

    return update_mask, obs_mask
