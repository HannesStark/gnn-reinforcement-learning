from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import numpy as np


def relation_matrix_to_adjacency_matrix(relation_matrix: List[List],
                                        one_hot_attributes: bool = True,
                                        self_loop: bool = True):

    with torch.no_grad():
        edge_matrix = torch.tensor(relation_matrix, dtype=np.long)

        if self_loop:
            edge_matrix += torch.diag(torch.ones(edge_matrix.shape[0],
                                                 dtype=edge_matrix.dtype))
        edge_matrix = edge_matrix.to_sparse()

        edge_indices = edge_matrix._indices()
        if one_hot_attributes:
            edge_attr = torch.nn.functional.one_hot(edge_matrix.values().abs())
            edge_attr += 1
        else:
            edge_attr = edge_matrix.values()[:, None]
    return edge_indices, edge_attr


def observations_to_node_attributes(observations: torch.Tensor,
                                    obs_input_mapping: dict,
                                    static_input_mapping: dict,
                                    num_nodes,
                                    num_node_features):

    batch_size, observation_sample_size = observations.shape
    # check that there is a mapping for every element of the observation vector
    assert max(list(obs_input_mapping.values()))[
        0] + 1 == observation_sample_size

    static_input_mapping = {int(k): np.concatenate(list(v.values()))
                            for k, v in static_input_mapping.items()}
    static_input_mapping[0] = np.array([])

    with torch.no_grad():
        attributes = torch.zeros(batch_size,
                                 num_nodes,
                                 num_node_features,
                                 dtype=torch.float32,
                                 device=observations.device)

        for i in range(num_nodes):
            in_size = len(obs_input_mapping[i])
            static_in_size = len(static_input_mapping[i])

            in_mask = list(range(in_size))
            static_in_mask = list(range(in_size, in_size+static_in_size))

            attributes[:, i, in_mask] = observations[:, obs_input_mapping[i]]
            attributes[:, i, static_in_mask] = torch.from_numpy(
                static_input_mapping[i]).expand(batch_size, -1).float().to(observations.device)

    return attributes
