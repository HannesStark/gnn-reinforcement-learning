from typing import Optional, Tuple, Dict, List, Union

import torch
from stable_baselines3.common.utils import get_device
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptTensor, PairTensor

from NerveNet.models.utils import glorot, zeros


class NerveNetConv(MessagePassing):
    r"""

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 cached: bool = False,
                 bias: bool = True,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(NerveNetConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_masks = update_masks
        self.use_bias = bias
        self.cached = cached
        self.device = get_device(device)

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.update_models_parameter = {}
        for group_name, _ in update_masks.items():
            self.update_models_parameter[group_name] = {}
            self.update_models_parameter[group_name]["weights"] = Parameter(
                torch.Tensor(in_channels, out_channels)).to(self.device)

            if self.use_bias:
                self.update_models_parameter[group_name]["bias"] = Parameter(
                    torch.Tensor(out_channels)).to(self.device)
            else:
                self.update_models_parameter[group_name]["bias"] = None

        self.reset_parameters()

    def reset_parameters(self):
        for _, params in self.update_models_parameter.items():
            glorot(params["weights"])
            zeros(params["bias"])
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """

        """

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        embedding = torch.zeros(
            (*inputs.shape[:-1], self.out_channels)).to(self.device)

        for group_name, (update_mask, _) in self.update_masks.items():
            masked_inputs = inputs[:, update_mask]
            embedding[:, update_mask] = torch.matmul(
                masked_inputs,
                self.update_models_parameter[group_name]["weights"])
            if self.use_bias:
                embedding[:, update_mask] += self.update_models_parameter[group_name]["bias"]

        return embedding

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class NerveNetConv_v1(NerveNetConv):
    """
        A message passing schema for nervenet with "Ego- and Neighbor-embedding Separation" 
        as described in "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"
    """

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """

        """
        # remove self loops, so we only aggregate neighbours and can add the ego features seperately
        edge_index, _ = remove_self_loops(edge_index)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        aggregated_neighbors = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                              size=None)

        # combine ego and aggregated neighbours
        out = torch.cat([x, aggregated_neighbors], dim=-1)

        return out


class NerveNetConvGRU(GatedGraphConv):

    def __init__(self,
                 out_channels: int,
                 num_layers: int,
                 update_masks: Dict[str, Tuple[List[int], int]],
                 aggr: str = 'add',
                 bias: bool = True,
                 **kwargs):

        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Parameter(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        batch_size = x.shape[0]
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               size=None)
            x = self.rnn(m.view(-1, self.out_channels), x.view(-1,
                                                               self.out_channels)).view(batch_size, -1, self.out_channels)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
