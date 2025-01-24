from torch_scatter import scatter
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor
import numbers


class BN(nn.Module):
    def __init__(self, m, type='ln', scale=0, eps=1e-8):
        super().__init__()
        if type == 'ln':
            self.normalization = LayerNorm1d(m, scale=scale, eps=eps)
        elif type == 'gn':
            self.normalization = GraphNorm(m)
        else:
            NotImplementedError
        self.type_ = type

    def forward(self, x, graph):
        return self.normalization(x, graph)


class GraphNorm(nn.Module):
    '''
    Graph normalization https://arxiv.org/abs/2009.03294
    '''
    def __init__(self, channel=64, type='gn'):
        super(GraphNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channel))
        self.bias = nn.Parameter(torch.zeros(channel))
        if type == 'gn':
            self.mean_scale = nn.Parameter(torch.ones(channel))
        elif type == 'in':
            self.mean_scale = None
        else:
            raise NotImplementedError

    def forward(self, tensor, graph):
        batch_list = graph.batch_num_nodes()
        batch_size = batch_list.shape[0]
        batch_index = torch.arange(batch_size, device=tensor.device).repeat_interleave(batch_list)
        mean = scatter(tensor, batch_index, dim=0, reduce='mean').repeat_interleave(batch_list, dim=0)
        if self.mean_scale is None:
            sub = tensor - mean
        else:
            sub = tensor - mean * self.mean_scale
        std = (scatter(sub ** 2, batch_index, dim=0, reduce='mean') + 1e-6).sqrt().repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

class LayerNorm1d(nn.Module):
    '''
    Just layer normalization. scale=1: skip this layer.
    '''
    def __init__(self, normalized_shape, eps: float = 1e-5, device=None, dtype=None, scale=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.scale = scale
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.bias = Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))

    def forward(self, input: Tensor, *args) -> Tensor:
        if self.scale == 0:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            return input