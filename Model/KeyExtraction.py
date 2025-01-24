import pdb
import torch
import torch.nn as nn
from Model.fiber_new import Tensor_fiber
import einops
import dgl
from torch_cluster import knn
from ModelNet40.data_utils import get_basis_new


class KeyExtractionLayer(nn.Module):
    def __init__(self, key_regress, key_extract, key_feature_gloabal_pool_weight, key_mean_num=10, max_deg=1,
                 key_global_merge='add_0'):
        super().__init__()
        self.key_regress = key_regress
        self.key_extract = key_extract
        self.key_mean_num = key_mean_num
        self.max_deg = max_deg
        self.key_feature_gloabal_pool_weight = key_feature_gloabal_pool_weight
        self.global_merge = key_global_merge

    def __repr__(self):
        return f"Key_point_layer"

    def forward(self, feature, graph, **kwargs):
        #############
        if not feature.fiber.can_fuse:
            raise NotImplementedError


        with graph.local_scope():
            graph.ndata['all_feature'] = feature.tensor

            logit, _ = self.key_feature_gloabal_pool_weight(feature, G=graph)

            assert logit.get_component((0, 0)).shape[1] > 1

            # averaged feature
            graph.ndata['logit'] = logit.get_component((0, 0))[:, 0:1, :]
            graph.ndata['prob'] = dgl.softmax_nodes(graph, 'logit')
            batched_tensor = dgl.sum_nodes(graph, 'all_feature', 'prob')

            # averaged degree 0 feature
            graph.ndata['global_0'] = logit.get_component((0, 0))[:, 1:, :]
            global_deg0 = dgl.sum_nodes(graph, 'global_0', 'prob')

            bs = batched_tensor.shape[0]
            tensor_1 = batched_tensor[0:bs // 2]
            tensor_2 = batched_tensor[bs // 2:]

            if self.global_merge == 'cat_0':
                # the degree 0 features are concat, and other features are kept
                global_deg0_1 = global_deg0[0:bs // 2]
                global_deg0_2 = global_deg0[bs // 2:]

                extra_paired_feature_deg0 = torch.cat([torch.cat([global_deg0_1, global_deg0_2], 1),
                                                           torch.cat([global_deg0_2, global_deg0_1], 1)], 0)

                extra_feature_pooled_deg0 = dgl.broadcast_nodes(graph, extra_paired_feature_deg0)
                paired_feature_deg0 = torch.cat([torch.cat([tensor_1[:, :, 0:1], tensor_2[:, :, 0:1]], 1),
                                                 torch.cat([tensor_2[:, :, 0:1], tensor_1[:, :, 0:1]], 1)], 0)

                feature_pooled_deg0 = dgl.broadcast_nodes(graph, paired_feature_deg0)

                feature_pooled_nonzero = dgl.broadcast_nodes(graph, batched_tensor[:, :, 1:])
                assert len(feature.fiber.keys()) == 2, 'Not Implement.'
                feature_new_dict = {(0, 0): torch.cat([feature.get_component((0,0)), extra_feature_pooled_deg0, feature_pooled_deg0], 1),
                                    (1, 0): torch.cat([feature.get_component((1,0)), feature_pooled_nonzero], 1)}
                feature_new = Tensor_fiber(tensor=feature_new_dict)
            else:
                raise NotImplementedError
            logit, graph = self.key_regress(feature_new, G=graph)

        # get the key point as a convex combination of the raw net
        with graph.local_scope():
            graph.ndata['pos_weight_logit'] = logit.get_component((0, 0))
            graph.ndata['pos_weight'] = dgl.softmax_nodes(graph, 'pos_weight_logit')
            graph.ndata['pos_reshape'] = graph.ndata['pos'].unsqueeze(1)
            selected_key_points = dgl.sum_nodes(graph, 'pos_reshape', 'pos_weight')

        ## select key point and build a biparti_graph to compute the features of the key points.
        num_keypoint = selected_key_points.shape[1]
        keypoints = einops.rearrange(selected_key_points, 'b n d -> (b n) d')

        Index = torch.arange(0, bs, device='cuda')
        Batch_al = torch.repeat_interleave(Index, graph.batch_num_nodes())
        Batch_se = torch.repeat_interleave(Index, num_keypoint)

        index = knn(graph.ndata['pos'], keypoints, self.key_mean_num, Batch_al, Batch_se)
        index_np = index.cpu().numpy()

        biparti_graph = dgl.heterograph({('U', 'E', 'V'): (index_np[1], index_np[0])})

        biparti_graph.nodes['U'].data['index'] = torch.arange(biparti_graph.num_nodes(ntype='U'))
        iso_index = (biparti_graph.out_degrees() == 0).nonzero().squeeze(1)
        biparti_graph.remove_nodes(iso_index, ntype='U')
        biparti_graph = biparti_graph.to('cuda')

        ## copy position from the original graph to the biparti graph
        biparti_graph.nodes['U'].data['pos'] = graph.ndata['pos'][biparti_graph.nodes['U'].data['index']]
        biparti_graph.nodes['V'].data['pos'] = keypoints

        ## get new homo graph, edge target: key points, edge source: knn neighbors of key points
        homo_graph = dgl.to_homogeneous(biparti_graph, ndata=['pos'])

        # compute basis. Get ready for message passing
        get_basis_new(self.max_deg, homo_graph)
        assert biparti_graph.ntypes[0] == 'U' and biparti_graph.ntypes[1] == 'V', 'Node shape inconsistent'

        # fill the key point feature by 1
        if not feature.can_fuse:
            New_f = {}
            for k in feature.fiber.keys():
                feature_all = feature.get_component('k')
                U_feature = feature_all[biparti_graph.nodes['U'].data['index']]
                V_feature = torch.zeros((biparti_graph.num_nodes(ntype='V'), feature_all.shape[1], feature_all.shape[2]), device='cuda')
                UV_feature = torch.cat([U_feature, V_feature], 0)
                New_f[k] = UV_feature
            new_feature = Tensor_fiber(tensor=New_f)
        else:
            feature_all = feature.tensor
            U_feature = feature_all[biparti_graph.nodes['U'].data['index']]
            V_feature = torch.zeros((biparti_graph.num_nodes(ntype='V'), feature_all.shape[1], feature_all.shape[2]), device='cuda')
            UV_feature = torch.cat([U_feature, V_feature], 0)
            new_feature = Tensor_fiber(fiber=feature.fiber, tensor=UV_feature)

        # do one step message passing
        feature_key_points_all, graph = self.key_extract(new_feature, G=homo_graph)

        # only keep the features of the key points
        if not feature_key_points_all.can_fuse:
            New_f_key = {}
            for k in feature.fiber.keys():
                New_f_key[k] = feature_key_points_all[k][-biparti_graph.num_nodes(ntype='V'):, :, :]
            key_tensor = Tensor_fiber(tensor=New_f_key)
        else:
            key_tensor = Tensor_fiber(fiber=feature_key_points_all.fiber, tensor=feature_key_points_all.tensor[-biparti_graph.num_nodes(ntype='V'):, :, :])

        # remove all original points and only keep key points in graph
        delete_index = (graph.ndata['_TYPE'] != 1).nonzero().squeeze(1)
        graph.remove_nodes(delete_index)

        graph.set_batch_num_nodes(torch.tensor([num_keypoint] * bs))
        graph.set_batch_num_edges(torch.tensor([0] * bs))
        return key_tensor, graph