import pdb
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
import einops
from .fiber_new import Fiber
from Model.MyBatchNorm import BN
import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from escnn.kernels.harmonic_polynomial_r3 import HarmonicPolynomialR3Generator
from Model.fiber_new import Tensor_fiber


def MergeFeatureCat(feature_batch, merge_type):
    size = feature_batch.get_component((0, 0)).shape[0] // 2
    New_feature_dict = {}
    if merge_type == '11':
        for deg in feature_batch.fiber.keys():
            if deg[1] == 0 and deg[0] == 0:
                # add degree-(0, 0) feature
                g1_side = feature_batch.get_component((0, 0))[:size]
                g2_side = feature_batch.get_component((0, 0))[size:]
                feature = g1_side + g2_side
                New_feature_dict[deg] = feature
            elif deg[0] == 1 and deg[1] == 0:
                # keep degree (0, 1) and degree (1, 0) features
                feature1 = feature_batch.get_component((deg[0], 0))[:size]
                New_feature_dict[(deg[0], 0)] = feature1
                feature2 = feature_batch.get_component((deg[0], 0))[size:]
                New_feature_dict[(0, deg[0])] = feature2
                # multiply degree (0, 1) and degree (1, 0) features to get degree (1, 1) feature, 3D x 3D = 9D
                New_feature_dict[(1, 1)] = einops.rearrange(einops.einsum(feature1, feature2, 'p c i, p c j -> p c i j'),
                                                            'p c i j -> p c (i j)')
            else:
                continue
        combined_feature = Tensor_fiber(tensor=New_feature_dict)
        return combined_feature
    else:
        raise NotImplementedError


def get_r_new(G):
    """Compute internodal distances"""
    if 'd' not in G.edata.keys():
        G.edata['d'] = G.ndata['pos'][G.edges()[1]] - G.ndata['pos'][G.edges()[0]]
    if G.edata['d'].shape[-1] == 6:
        d1 = torch.linalg.vector_norm(G.edata['d'][:, 0:3], dim=-1, keepdim=True)
        d2 = torch.linalg.vector_norm(G.edata['d'][:, 3:6], dim=-1, keepdim=True)
        return torch.cat([d1, d2], -1)
    elif G.edata['d'].shape[-1] == 3:
        d1 = torch.linalg.vector_norm(G.edata['d'][:, 0:3], dim=-1, keepdim=True)
        return d1


def get_basis_new(max_degree, max_degree_side, Q_cache, G):
    '''2nd order CG coefficient.'''
    device = G.device
    harmonic_max_degree = max(min(max_degree, max_degree_side[0]), min(max_degree, max_degree_side[1]))
    harmonics_generator = HarmonicPolynomialR3Generator(2*harmonic_max_degree).to(device)

    if G.edata['d'].shape[-1] == 6:
        sphere1 = torch.nn.functional.normalize(G.edata['d'][:, :3], dim=1)
        sphere2 = torch.nn.functional.normalize(G.edata['d'][:, 3:], dim=1)
        Y_escnn2 = harmonics_generator(sphere2)
    elif G.edata['d'].shape[-1] == 3:
        sphere1 = torch.nn.functional.normalize(G.edata['d'][:, :3], dim=1)
        Y_escnn2 = torch.ones((G.edata['d'].shape[0], 1), device=device)
    Y_escnn1 = harmonics_generator(sphere1)


    Y12_all_new = {}
    Y12escnn = einops.einsum(Y_escnn1, Y_escnn2, 'b i, b j -> i j b')
    for J1 in range(min(max_degree * 2 + 1, max_degree_side[0] * 2 + 1)):
        for J2 in range(min(max_degree * 2 + 1, max_degree_side[1] * 2 + 1)):
            Y12_all_new[J1, J2] = einops.rearrange(Y12escnn[J1 ** 2: (J1 + 1) ** 2, J2 ** 2: (J2 + 1) ** 2, :], 'i j b -> 1 b (i j)')


    basis_new = {}
    for in1 in range(min(max_degree + 1, max_degree_side[0] + 1)):
        for in2 in range(min(max_degree - in1 + 1, max_degree_side[1] + 1)):
            for out1 in range(min(max_degree + 1, max_degree_side[0] + 1)):
                for out2 in range(min(max_degree - out1 + 1, max_degree_side[1] + 1)):
                    escnn_Q = Q_cache.get_Q(in1, in2, out1, out2)
                    size_out = (2 * out1 + 1) * (2 * out2 + 1)

                    Y12_list = []
                    block_size_list = []
                    for J1 in range(abs(in1-out1), in1+out1+1):
                        for J2 in range(abs(in2-out2), in2+out2+1):
                            block_size = (2 * J1 + 1) * (2 * J2 + 1)
                            Y12_list.append(Y12_all_new[J1, J2])
                            block_size_list.append(block_size)

                    block_size_tensor = torch.arange(0, len(block_size_list), device=device)
                    index = torch.repeat_interleave(block_size_tensor, torch.tensor(block_size_list, device=device))
                    escnn_Q_cuda = torch.tensor(escnn_Q, dtype=torch.float, device=device)
                    dotprod = escnn_Q_cuda.unsqueeze(1) * torch.cat(Y12_list, -1)
                    K_J_all = scatter(dotprod, index, dim=-1)
                    new_basis_K = einops.rearrange(K_J_all, '(i o) p f -> p 1 o 1 i f', o=size_out)
                    basis_new[f'(({in1}, {in2}),({out1}, {out2}))'] = new_basis_K

    return basis_new

def get_basis_and_r_new(max_degree, Q_cache, g=None, max_degree_side=None):
    '''Compute CG coefficient and distance for grraph G.'''
    r_all = get_r_new(g)
    basis = get_basis_new(max_degree, max_degree_side, Q_cache, g)

    return basis, r_all


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""
    def __init__(self, num_freq, in_dim, out_dim, edge_dim: int=0, symmetric=False, scale_deg=0):
        """NN parameterized radial profile function.

        """
        super().__init__()

        self.num_freq = num_freq[0] * num_freq[1]
        self.num_freq_component = num_freq

        self.in_dim = in_dim
        self.mid_dim = 32
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.symmetric = symmetric

        # if scale equivariant, no bias; if scale invariant, then use bias
        if scale_deg == 1:
            use_bias = False
        else:
            use_bias = True
        self.scale_deg = scale_deg

        self.net = nn.ModuleList([nn.Linear(self.edge_dim, self.mid_dim, bias=False),
                                 #########
                                 BN(self.mid_dim, type='ln', scale=scale_deg),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Linear(self.mid_dim, self.mid_dim, bias=use_bias),
                                 BN(self.mid_dim, type='ln', scale=scale_deg),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Linear(self.mid_dim, self.num_freq * in_dim * out_dim, bias=use_bias)])

        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.kaiming_uniform_(self.net[3].weight)
        nn.init.kaiming_uniform_(self.net[6].weight)

    def __repr__(self):
        return f"RadialFunc(edge_dim={self.edge_dim}, in_dim={self.in_dim}, out_dim={self.out_dim})"

    def run_net(self, x, graph):
        for layer in self.net:
            if type(layer) == BN:
                x = layer(x, graph)
            else:
                x = layer(x)
        return x

    def forward(self, x, reverse=False, graph=None):
        pos_encode = x

        if self.edge_dim == 1:
            y = self.run_net(pos_encode, graph)
        else:
            if reverse:
                reverse_pos_encode = torch.cat([pos_encode[:, 1:], pos_encode[:, 0:1]], 1)
                y = self.run_net(reverse_pos_encode, graph)
                y = einops.rearrange(y, 'p (o i a b) -> p (o i b a)', o=self.out_dim,
                                     i=self.in_dim, a=self.num_freq_component[0], b=self.num_freq_component[1])
            elif self.symmetric:
                reverse_pos_encode = torch.cat([pos_encode[:, 1:],  pos_encode[:, 0:1]], 1)
                y_order = self.run_net(pos_encode, graph)
                y_reverse = self.run_net(reverse_pos_encode, graph)
                y_reverse = einops.rearrange(y_reverse, 'p (o i a b) -> p (o i b a)', o=self.out_dim,
                                     i=self.in_dim, a=self.num_freq_component[0], b=self.num_freq_component[1])
                y = (y_order + y_reverse) * 0.5
            else:
                y = self.run_net(pos_encode, graph)
        out = y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)
        return out


class KernelComputeFun(nn.Module):
    ''' Compute kernel as the product of radial function and basis.'''
    def __init__(self, rp, reverse=False):
        super().__init__()
        self.rp = rp
        self.reverse = reverse

    def forward(self, feat, basis, graph):
        R = self.rp(feat, reverse=self.reverse, graph=graph)
        kernel = torch.sum(R * basis, -1)
        return kernel
    ############

class ConvPartial(nn.Module):
    """Equivariant Graph convolution"""
    def __init__(self, f_in, f_out, edge_dim: int=0, scale_deg=0):

        super().__init__()
        self.f_out = f_out
        self.edge_dim = edge_dim
        self.f_in = f_in

        # define rp outside
        self.rp = nn.ModuleDict()
        for di, mi in self.f_in.items():
            for do, mo in self.f_out.items():
                num_freq = (2 * min(di[0], do[0]) + 1, 2 * min(di[1], do[1]) + 1)
                if edge_dim == 1:
                    self.rp[f'({di},{do})'] = RadialFunc(num_freq, mi, mo, edge_dim, scale_deg=scale_deg)
                else:
                    if f'({(di[1], di[0])},{do[1], do[0]})' not in self.rp.keys():
                        symmetric = (di[1] == di[0]) and (do[1] == do[0])
                        # for symmetric degree, radial functions should also be symmetric. See Eqn 74 for details.
                        self.rp[f'({di},{do})'] = RadialFunc(num_freq, mi, mo, edge_dim, symmetric=symmetric,
                                                            scale_deg=scale_deg)

        self.kernel_unary = nn.ModuleDict()
        for di, mi in self.f_in.items():
            for do, mo in self.f_out.items():
                if f'({di},{do})' in self.rp.keys():
                    self.kernel_unary[f'({di},{do})'] = KernelComputeFun(self.rp[f'({di},{do})'])
                else:
                    # To keep swap equivariance, kernels should be symmetric See. Proposition 5.2
                    self.kernel_unary[f'({di},{do})'] = KernelComputeFun(self.rp[f'({di[1], di[0]},{do[1], do[0]})'], reverse=True)


    def __repr__(self):
        return f'ConvPartial(structure={self.f_out})'

    def udf_u_mul_e(self, d_out):
        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for d_in, m_in in self.f_in.items():
                msg = msg + torch.einsum('eabij,eij->eab', edges.data[f'({d_in},{d_out})'], edges.src[d_in])
            return {f'out{d_out}': msg}
        return fnc

    def udf_u_mul_e_full(self):
        def fnc(edges):
            msg = torch.einsum('eabij,eij->eab', edges.data['all_kernel'], edges.src['all_tensor'])
            return {'all': msg}
        return fnc


    # @profile
    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            h: dict of node-features
            G: minibatch of (homo)graphs
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """

        debug_info = {}
        with G.local_scope():
            # Add node features to local graph scope
            ###########
            for di, mi in self.f_in.items():
                for do, mo in self.f_out.items():
                    etype = f'({di},{do})'
                    G.edata[etype] = self.kernel_unary[etype](G.edata['r'], G.edata[f'basis_{etype}'], G)


            if self.f_in.can_fuse and self.f_out.can_fuse:
                Dict_all = []
                for do, mo in self.f_out.items():
                    Dict_out = []
                    for di, mi in self.f_in.items():
                        Dict_out.append(G.edata[f'({di},{do})'])
                    Dict_all.append(torch.cat(Dict_out, -1))
                Kernel_all = torch.cat(Dict_all, 2)
                G.edata['all_kernel'] = Kernel_all
                G.ndata['all_tensor'] = h.tensor
                G.apply_edges(self.udf_u_mul_e_full())
                Tensor_out = Tensor_fiber(fiber=self.f_out, tensor=G.edata['all'])
                return Tensor_out, debug_info

            for k, v in h.get_all_component().items():
                G.ndata[k] = v

            Out_dict = {}
            for d in self.f_out.keys():
                G.apply_edges(self.udf_u_mul_e(d))
                Out_dict[d] = G.edata[f'out{d}']

            return Tensor_fiber(tensor=Out_dict), debug_info


class Atten(nn.Module):
    """An equivariant multi-headed self-attention module for DGL graphs."""
    def __init__(self, f_value, f_key, n_heads: int):
        """
        Args:
            f_value: Fiber() object for value-embeddings
            f_key: Fiber() object for key-embeddings
            n_heads: number of heads
        """
        super().__init__()
        self.f_value = f_value
        self.f_key = f_key
        self.n_heads = n_heads


    def __repr__(self):
        return f'Att(n_heads={self.n_heads}, structure={self.f_value})'

    def udf_u_mul_e(self, d_out):
        def fnc(edges):
            # Neighbor -> center messages
            attn = edges.data['a']
            value = edges.data[f'v{d_out}']
            msg = attn.unsqueeze(-1).unsqueeze(-1) * value

            return {'m': msg}
        return fnc

    def udf_u_mul_e_full(self):
        def fnc(edges):
            # Neighbor -> center messages
            attn = edges.data['a']
            value = edges.data[f'v_all']
            msg = attn.unsqueeze(-1).unsqueeze(-1) * value
            return {'m': msg}
        return fnc


    def forward(self, v, k, q, G=None, **kwargs):
        """Message passing.
        Always try to pass all degrees in a fused way. If not possible, then pass each degree independently.

        Args:
            G: minibatch of graphs
            v: dict of value edge-features
            k: dict of key edge-features
            q: dict of query node-features
        """

        with G.local_scope():
            if v.can_fuse:
                assert v.tensor.shape[1] // self.n_heads > 0, 'insufficient channel for multi-head'
                G.edata['v_all'] = einops.rearrange(v.tensor, 'b (h n) d -> b h n d', h=self.n_heads)
            else:
                for d, m in self.f_value.items():
                    G.edata[f'v{d}'] = einops.rearrange(v.get_component(d), 'b (h n) d -> b h n d', h=self.n_heads)

            # compute attention
            k_head = k.sep_head(self.n_heads)
            q_head = q.sep_head(self.n_heads)
            e = einops.einsum(k_head, q_head[G.edges()[1]], 'e s t, e s t -> e s') / np.sqrt(self.f_key.deg_size)
            G.edata['a'] = edge_softmax(G, e)

            if v.can_fuse:
                G.update_all(self.udf_u_mul_e_full(), fn.sum('m', 'out_all'))
                out_tensor = einops.rearrange(G.ndata[f'out_all'], 'b h n d -> b (h n) d', h=self.n_heads)
                Tensor_out = Tensor_fiber(fiber=self.f_value, tensor=out_tensor)
            else:
                for d in self.f_value.keys():
                    G.update_all(self.udf_u_mul_e(d), fn.sum('m', f'out{d}'))
                output = {}
                for d, m in self.f_value.items():
                    output[d] = einops.rearrange(G.ndata[f'out{d}'], 'b h n d -> b (h n) d', h=self.n_heads)
                Tensor_out = Tensor_fiber(tensor=output)

        return Tensor_out, G

class EncodeLayer(nn.Module):
    """Equivariant Graph attention block"""
    def __init__(self, f_in, f_out, edge_dim: int=0, div: int=4, n_heads: int=1, bias=False, scale_deg: int=0):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.div = div
        self.n_heads = n_heads

        # f_mid_out has same structure as 'f_out' but #channels divided by 'div'. this will be used for the values
        for k, v in self.f_out.items():
            assert v // div > 0, f'Incorrect value channel setting in deg {k}'
        f_mid_out = {k: int(v // div) for k, v in self.f_out.items()}
        self.f_mid_out = Fiber(fiber=f_mid_out)

        # f_mid_in has same structure as f_mid_out, but only degrees which are in f_in
        # this will be used for keys and queries (queries are merely projected, hence degrees have to match input)
        f_mid_in = {d: m for d, m in f_mid_out.items() if d in self.f_in.keys()}
        self.f_mid_in = Fiber(fiber=f_mid_in)

        self.edge_dim = edge_dim
        self.scale_deg = scale_deg

        self.GMAB = nn.ModuleDict()
        if scale_deg == 0:
            # For scale invariance, v and k are both scale invariant and can be computed together
            self.GMAB['vk'] = ConvPartial(f_in, self.f_mid_out.cat(self.f_mid_in, complete=True), edge_dim=edge_dim,
                                         scale_deg=0)
        else:
            # for degree-1 scale equivariant. v is degree 1, and k is degree k.
            self.GMAB['v'] = ConvPartial(f_in, self.f_mid_out, edge_dim=edge_dim, scale_deg=scale_deg)
            self.GMAB['k'] = ConvPartial(f_in, self.f_mid_in, edge_dim=edge_dim, scale_deg=0)

        self.GMAB['q'] = GLinear(f_in, self.f_mid_in, bias=bias)
        self.GMAB['attn'] = Atten(self.f_mid_out, self.f_mid_in, n_heads=n_heads)

        #########################
        if scale_deg == 0:
            self.project = nn.Sequential(
                nn.Identity(),
                GLinear(self.f_mid_out.cat(self.f_in), f_out, bias=bias))
        elif scale_deg == 1:
            # use a linear layer to keep degree-1 scale equivariant
            self.project = nn.Sequential(
                GLinear(self.f_mid_out, f_out, bias=bias, use_skip=False))
        else:
            raise ValueError('linear_cat type not known')


    def forward(self, features, G, **kwargs):
        # Embeddings
        assert features.fiber.structure == self.f_in.structure, 'Input shape check failed.'

        if 'vk' in self.GMAB.keys():
            vk, conv_debug = self.GMAB['vk'](features, G=G, **kwargs)
            v, k = vk.get_sub_tensor(self.f_mid_in)
        else:
            k, conv_debug_k = self.GMAB['k'](features, G=G, **kwargs)
            v, conv_debug_v = self.GMAB['v'](features, G=G, **kwargs)
        q = self.GMAB['q'](features, G=G)

        z, g_new = self.GMAB['attn'](v, k=k, q=q, G=G)

        if self.scale_deg == 0:
            z = z.cat_tensor(features)
            out = self.project(z)
        elif self.scale_deg == 1:
            out = self.project(z)

        return out, g_new


class GLinear(nn.Module):
    '''
    A linear layer on equivariant graph. No mixed degree computation.
    '''
    def __init__(self, f_in, f_out, bias=False, use_skip=True):
        """SE(3)-equivariant 1x1 convolution.

        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.use_skip = use_skip
        self.transform = nn.ParameterDict()

        trans_deg = {}
        for do, mo in self.f_out.items():
            if do in self.f_in.keys():
                trans_deg[do] = mo
                if str((do[1], do[0])) in self.transform.keys():
                    self.transform[str(do)] = self.transform[str((do[1], do[0]))]
                else:
                    mi = self.f_in.structure[do]
                    self.transform[str(do)] = nn.Parameter(torch.randn(mo, mi) / np.sqrt(mi))
        self.f_trans = Fiber(trans_deg)

    def __repr__(self):
         return f"GLinear(structure={self.f_out})"

    def forward(self, features, **kwargs):
        assert features.fiber.structure == self.f_in.structure, 'Input shape check failed.'

        # If the tensors can be fused
        if self.f_out.can_fuse and self.f_in.can_fuse:
            stack_para = torch.stack([einops.rearrange(self.transform[str(k)], 'o i -> i o') for k in self.f_trans.keys()])
            stacked_feature = torch.cat([einops.rearrange(features.get_component(k), 'p c d -> (p d) c') for k in self.f_trans.keys()], 0)
            stack_len = torch.tensor([(2 * k[0] + 1) * (2 * k[1] + 1) for k in self.f_trans.keys()]) * features.tensor.shape[0]
            out_list = torch.split(dgl.ops.segment_mm(stacked_feature, stack_para, stack_len), stack_len.tolist(), dim=0)
            Out = torch.cat([einops.rearrange(i, '(p d) c -> p c d', p=features.tensor.shape[0]) for i in out_list], -1)
            Linear_tensor = Tensor_fiber(fiber=self.f_trans, tensor=Out)
            if self.use_skip and self.f_in.shared_channel == self.f_out.shared_channel:
                feature_in_trans = torch.cat([v for k, v in features.get_all_component().items() if k in self.f_trans.keys()], -1)
                Linear_tensor.tensor += feature_in_trans
            return Linear_tensor

        # If the tensors can not be fused, each degree is processed independently.
        output = {}
        #########
        for k, v in features.get_all_component().items():
            if str(k) in self.transform.keys():
                # use skip connection if possible
                if self.use_skip and self.f_in.structure[k] == self.f_out.structure[k]:
                    output[k] = torch.matmul(self.transform[str(k)], v) + v
                else:
                    output[k] = torch.matmul(self.transform[str(k)], v)
        return Tensor_fiber(tensor=output)