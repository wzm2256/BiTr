import pdb
import dgl
import torch
import torch.nn as nn
from Model.fiber_new import Tensor_fiber, Fiber
import einops
from torch_scatter import scatter


class GProjRelu(nn.Module):
    def __init__(self, fiber, eps=1e-8, param_type=2, slope=0.2):
        super().__init__()
        self.fiber = fiber
        self.eps = eps

        self.param_dict = nn.ParameterDict()

        self.f_non_zero = Fiber(fiber={d:m for (d,m) in self.fiber.structure.items() if d !=(0,0)})
        for d, m in fiber.items():
            if d == (0,0):
                # degree (0,0) is non-parametric
                continue

            # To keeo swap equivariance, nonlinear layer needs to be symmetric. See Prop 5.2 (2).
            if f'plane_{(d[1], d[0])}' in self.param_dict.keys():
                self.param_dict[f'plane_{d}'] = self.param_dict[f'plane_{(d[1], d[0])}']
            else:
                if param_type == 1:
                    self.param_dict.update({f'plane_{d}': nn.Parameter(torch.randn(1, m) / (m ** 2))})
                elif param_type == 2:
                    self.param_dict.update({f'plane_{d}': nn.Parameter(torch.randn(m, m) / m)})

        self.slope = slope
        self.type = param_type

    def __repr__(self):
        return f"GProjRelu()"

    def forward(self, tensor_fiber, **kwargs):

        if self.fiber.can_fuse:
            non_zero_para = torch.stack([einops.rearrange(self.param_dict[f'plane_{k}'], 'o i -> i o') for k in self.f_non_zero.keys()])
            non_zero_feature = torch.cat([einops.rearrange(tensor_fiber.get_component(k), 'p c d -> (p d) c') for k in self.f_non_zero.keys()], 0)
            dim_list = torch.tensor([(2 * k[0] + 1) * (2 * k[1] + 1) for k in self.f_non_zero.keys()], device='cuda')

            stack_len = dim_list * tensor_fiber.tensor.shape[0]
            out_list = torch.split(dgl.ops.segment_mm(non_zero_feature, non_zero_para, stack_len.to('cpu')), stack_len.tolist(), dim=0)
            d_all_tensor = torch.cat([einops.rearrange(i, '(p d) c -> p c d', p=tensor_fiber.tensor.shape[0]) for i in out_list], -1)

            non_zero_x_tensor = torch.cat([tensor_fiber.get_component(k) for k in self.f_non_zero.keys()], -1)

            dim_index = torch.repeat_interleave(torch.arange(0, len(dim_list), device='cuda'), dim_list)
            dotprod_all = scatter(d_all_tensor * non_zero_x_tensor, dim_index, dim=-1, reduce='sum')
            mask_all = (dotprod_all >= 0).float()
            d_norm_all = scatter(d_all_tensor * d_all_tensor, dim_index, dim=-1, reduce='sum').clamp_min(self.eps ** 2) ** 0.5
            mask_origin = torch.repeat_interleave(mask_all, dim_list, 2)
            d_norm_all_origin = torch.repeat_interleave(d_norm_all, dim_list, 2)
            dotprod_all_origin = torch.repeat_interleave(dotprod_all, dim_list, 2)

            relu_x_all = mask_origin * non_zero_x_tensor + (1 - mask_origin) * (non_zero_x_tensor - d_all_tensor * (dotprod_all_origin / d_norm_all_origin))
            x_out_nonzero = self.slope * non_zero_x_tensor + (1 - self.slope) * relu_x_all

            # handel 0 degree
            x_out_zero = nn.LeakyReLU(negative_slope=self.slope)(tensor_fiber.get_component((0, 0)))
            x_out = torch.cat([x_out_zero, x_out_nonzero], -1)

            return Tensor_fiber(tensor=x_out, fiber=self.fiber), None

        output = {}
        for deg, x in tensor_fiber.get_all_component().items():
            if deg == (0, 0):
                output[deg] = nn.LeakyReLU(negative_slope=self.slope)(x)
            elif deg != (0,0):
                d = einops.einsum(self.param_dict[f'plane_{deg}'], x, 't d, b d k -> b t k') # p c d
                dotprod = (x * d).sum(2, keepdim=True) # p c 1
                mask = (dotprod >= 0).float() # p c 1
                d_norm_sq = d.norm(dim=2, keepdim=True).clamp_min(self.eps) # p c 1
                relu_x = mask * x + (1 - mask) * (x - d * (dotprod / d_norm_sq))
                x_out_old = self.slope * x + (1 - self.slope) * relu_x
                output[deg] = x_out_old

        return Tensor_fiber(tensor=output), None