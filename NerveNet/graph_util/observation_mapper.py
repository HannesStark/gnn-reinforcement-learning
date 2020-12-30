import torch
import numpy as np


def observations_to_node_attributes(observations: torch.Tensor,
                                    obs_input_mapping: dict,
                                    static_input_mapping: dict):

    assert len(obs_input_mapping) == len(observations)
    assert len(static_input_mapping) + 1 == len(observations)

    static_input_mapping = {int(k): np.concatenate(list(v.values()))
                            for k, v in static_input_mapping.items()}
    static_input_mapping[0] = np.array([])

    # number of nodes
    n = len(obs_input_mapping)
    # node feature size
    d = max([len(lst1) + len(lst1)
             for lst1 in obs_input_mapping.values()
             for lst2 in static_input_mapping.values()])

    attributes = torch.zeros(n, d, dtype=torch.float32)

    for i in range(n):
        in_size = len(obs_input_mapping[i])
        static_in_size = len(static_input_mapping[i])
        in_mask = list(range(in_size))
        static_in_mask = list(range(in_size, in_size+static_in_size))
        attributes[i, in_mask] = observations[obs_input_mapping[i]]
        attributes[i, static_in_mask] = torch.from_numpy(
            static_input_mapping[i]).float()

    return attributes
