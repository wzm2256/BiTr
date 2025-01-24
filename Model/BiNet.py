import logging

from torch import nn
from .modules import EncodeLayer, get_basis_and_r_new
from Model.NonLinear import GProjRelu
from .fiber_new import Fiber
import compute_S
import numpy as np
from Model.KeyExtraction import KeyExtractionLayer
import dgl
from .modules import MergeFeatureCat
from Model.fiber_new import Tensor_fiber
from Model.Predict import Head
from ModelNet40.data_utils import catpc2_graph




class Q_cache:
    # a simple class for caching Q
    def __init__(self):
        self.cache = {}

    def get_Q(self, in1, in2, out1, out2):
        if (in1, in2, out1, out2) in self.cache.keys():
            return self.cache[(in1, in2, out1, out2)]
        else:
            escnn_Q, _ = compute_S.Q(in1, in2, out1, out2)
            self.cache.update({(in1, in2, out1, out2): escnn_Q})
            return escnn_Q



class BiNet(nn.Module):
    def __init__(self):
        super().__init__()



class BiSE3Transformer(BiNet):
    """Basic building block of BITR"""

    def __init__(self, Construct_fibers, num_layers: int, div: int = 2, edge_dim: int = 2,
                 n_heads: int = 1,
                 max_deg_total=None, max_degree_side=None,
                 num_key_points=-1,  key_mean_num=10,
                 binet=False,
                 key_global_merge='add_0', key_global_merge_coe=0,
                 ):

        super().__init__()
        self.num_layers = num_layers
        self.div = div
        self.n_heads = n_heads
        self.fibers = Construct_fibers

        self.edge_dim = edge_dim
        self.Q = Q_cache()

        self.num_key_points = num_key_points
        self.key_mean_num = key_mean_num
        self.binet = binet
        self.key_global_merge = key_global_merge
        self.key_global_merge_coe = key_global_merge_coe

        if max_deg_total is None:
            self.max_deg_total = 2
        else:
            self.max_deg_total = max_deg_total

        if max_degree_side is None:
            self.max_degree_side = (np.inf, np.inf)
        else:
            self.max_degree_side = max_degree_side

        self.Gblock, self.Read_out_layers, self.key_layer = self._build_gcn(self.fibers)
        print(self.Gblock)

    def _build_gcn(self, fibers):
        if self.num_key_points > 0:
            key_fiber = Fiber(fiber={(0, 0): self.num_key_points})
            feature_weight_fiber = Fiber(fiber={(0, 0): 1 + self.key_global_merge_coe})

            fout = fibers['mid']
            if self.key_global_merge == 'cat_0':
                fiber_global_dict = {d:2 * m for d,m in fibers['mid'].structure.items() if d !=(0,0)}
                fiber_global_dict[(0,0)] = 3 * fibers['mid'].structure[(0,0)] + 2 * self.key_global_merge_coe
                feature_fiber_new = Fiber(fiber=fiber_global_dict)
            else:
                raise NotImplementedError

            key_regress = EncodeLayer(feature_fiber_new, key_fiber, edge_dim=self.edge_dim, div=self.div,
                                      n_heads=self.n_heads)
            key_extract = EncodeLayer(fibers['mid'], fibers['out'], edge_dim=self.edge_dim, div=self.div,
                                      n_heads=self.n_heads)
            key_feature_gloabal_pool_weight = EncodeLayer(fibers['mid'], feature_weight_fiber, edge_dim=self.edge_dim,
                                                          div=1, n_heads=1)
            key_point_layer = KeyExtractionLayer(key_regress, key_extract, key_feature_gloabal_pool_weight,
                                                 key_mean_num=self.key_mean_num,
                                                 max_deg=max(self.max_deg_total, 1),
                                                 key_global_merge=self.key_global_merge)
        else:
            key_point_layer = None
            fout = fibers['out']

        Gblock = []
        fin = fibers['in']
        ReadOut_list = []

        for i in range(self.num_layers):
            Gblock.append(EncodeLayer(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads))
            Gblock.append(GProjRelu(fibers['mid']))
            fin = fibers['mid']

        if self.edge_dim == 2:
            # the bitr network is degree-1 scale equivariant
            scale_deg = 1
        else:
            # The feature extraction network is scale invariant
            scale_deg = 0

        Gblock.append(EncodeLayer(fin, fout, edge_dim=self.edge_dim, div=1, n_heads=1, scale_deg=scale_deg))

        return nn.ModuleList(Gblock), nn.ModuleList(ReadOut_list), key_point_layer


    def forward(self, feature_dict, g=None):
        if type(feature_dict) == dict:
            tensor_fiber = Tensor_fiber(tensor=feature_dict)
        else:
            tensor_fiber = feature_dict

        assert tensor_fiber.fiber.structure == self.fibers['in'].structure, f'Shape check fails: input shape: {tensor_fiber.fiber.structure}'

        if self.binet:
            max_deg = max(self.max_deg_total, 2)
        else:
            max_deg = max(self.max_deg_total, 1)

        if 'r' not in g.edata.keys():
            basis, r = get_basis_and_r_new(g=g, max_degree=max_deg, Q_cache=self.Q, max_degree_side=self.max_degree_side)
            for k in basis.keys():
                g.edata[f'basis_{k}'] = basis[k]
            g.edata['r'] = r

        for index, layer in enumerate(self.Gblock):
            tensor_fiber, g_new = layer(tensor_fiber, G=g)
            if g_new is not None:
                g = g_new

        if self.num_key_points > 0:
            feature, g_key = self.key_layer(tensor_fiber, g)
            return feature, g_key

        return tensor_fiber, g



class BiSE3TransformerDown(nn.Module):
    def __init__(self, feature_Construct_fibers, feature_num_layers, Construct_fibers, num_layers,
                div: int = 2,
                 n_heads: int = 1,
                 max_deg_total=None,
                 max_degree_side=None,
                 feature_max_deg_total=None,
                 merge_type=None,
                 num_head=4,
                 re_convert_k=2,
                 num_key_points=-1,
                 key_mean_num=10,
                 key_global_merge='cat_0',
                 key_global_merge_coe=0, predict_norm='gn',
                 ):
        super().__init__()
        self.feature_extract = BiSE3Transformer(feature_Construct_fibers, feature_num_layers,
                                                div=div, edge_dim=1, n_heads=n_heads,
                                                max_deg_total=feature_max_deg_total, max_degree_side=(np.inf, 0),
                                                num_key_points=num_key_points,
                                                key_mean_num=key_mean_num, binet=False,
                                                key_global_merge=key_global_merge,
                                                key_global_merge_coe=key_global_merge_coe)
        self.Binet = BiSE3Transformer(Construct_fibers, num_layers, div=div, edge_dim=2, n_heads=n_heads,
                                      max_deg_total=max_deg_total, max_degree_side=max_degree_side,
                                      num_key_points=-1,
                                      binet=True)

        self.merge_type = merge_type
        self.predict_head = Head(num_head=num_head, predict_norm=predict_norm)
        self.max_deg_total = max_deg_total

        self.logger = logging.getLogger('Network')
        self.re_convert_k = re_convert_k
        self.Construct_fibers = Construct_fibers

        self.num_key_points = num_key_points

    def forward(self, feature_dict, g=None):
        # etract features of both graph
        feature, graph = self.feature_extract(feature_dict=feature_dict, g=g)
        feature_dict_batch = MergeFeatureCat(feature, self.merge_type)

        # get a graph in 6D space
        graph_list = dgl.unbatch(graph)
        g1 = dgl.batch(graph_list[:len(graph_list) // 2])
        g2 = dgl.batch(graph_list[len(graph_list) // 2: ])
        cat_g = catpc2_graph(g1.ndata['pos'], g2.ndata['pos'], k_neighbor=self.re_convert_k, num_kp=self.num_key_points)


        # apply Binet in the 6D space
        tensor_fiber, g_final = self.Binet(feature_dict=feature_dict_batch, g=cat_g)
        r, t, r_vector = self.predict_head(tensor_fiber, g_final)

        return r, t, g1, g2, tensor_fiber, r_vector
