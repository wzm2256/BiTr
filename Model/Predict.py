import pdb

import torch
import torch.nn as nn
import dgl
import einops
from Model.modules import GLinear
from .fiber_new import Fiber, Tensor_fiber
from Model.MyBatchNorm import BN

def project_r(r_vector):
	'''
	SVD projection
	'''

	norm = torch.linalg.norm(r_vector, dim=1, keepdim=True).clamp_min(1e-5)
	r_vector = r_vector / norm.detach()
	r_ = einops.rearrange(r_vector, 'b (d c) -> b c d', c=3)

	U, S, Vh = torch.linalg.svd(r_, full_matrices=False)
	if S.norm() < 1e-4:
		print('Warning: Matrix is degraded, considering using denser graph')

	rot_mat_pos = U @ Vh
	Vh_neg = Vh.clone()
	Vh_neg[:, 2, :] *= -1
	rot_mat_neg = U @ Vh_neg
	rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	return rot_mat


class Head(nn.Module):
	'''
	The last linear layer with a permutation matrix, which is used to transform the result to the input coordinate frame/
	'''
	def __init__(self, num_head, predict_norm='gn'):
		super().__init__()

		self.Q_ = torch.tensor([[0., 0., 1.],
		                  [1., 0., 0.],
		                  [0., 1., 0.]])

		self.register_buffer('Q', self.Q_)
		self.register_buffer('Q_tensor', torch.kron(self.Q_, self.Q_))

		self.weight_logit_layer = nn.ModuleList([BN(num_head * 4, type=predict_norm),
												nn.Linear(num_head * 4, 3, bias=True)
												])

		self.predict_dict = {(0, 1): 1, (1, 0): 1, (1, 1): 1}
		self.predict_dict_head = {(0, 1): num_head, (1, 0): num_head, (1, 1): num_head}
		self.linear_head = GLinear(Fiber(fiber=self.predict_dict_head), Fiber(fiber=self.predict_dict), use_skip=False)

	def run_net(self, x, net, graph):
		for layer in net:
			if type(layer) == BN:
				x = layer(x, graph)
			else:
				x = layer(x)
		return x

	def forward(self, tensor_fiber, g, **kwargs):
		with g.local_scope():
			Head_dict = {}

			# compute invariant weights for each node
			weight_logit = self.run_net(tensor_fiber.get_component((0, 0))[:, :, 0], self.weight_logit_layer, g)
			g.ndata['weight_logits'] = weight_logit
			W = dgl.softmax_nodes(g, 'weight_logits')

			# compute degree (1, 0), (0, 1), (1, 1,) output as weighted average
			g.ndata['ab'] = torch.cat([tensor_fiber.get_component((1, 0)), tensor_fiber.get_component((0, 1))], -1)
			g.ndata['r'] = tensor_fiber.get_component((1, 1))
			for i in range(2, 3):
				g.ndata[f'weight_{i}'] = W[:, i:i+1]
			for i in range(2):
				g.ndata[f'weight_{i}'] = W[:, i:i+1].unsqueeze(-1)
			m1m2 = dgl.sum_nodes(g, 'pos', 'weight_2')
			mean_ab = dgl.sum_nodes(g, 'ab', 'weight_1')
			m1 = m1m2[:, 0:3]
			m2 = m1m2[:, 3:]
			Head_dict[(1, 0)] = mean_ab[:, :, :3]
			Head_dict[(0, 1)] = mean_ab[:, :, 3:6]
			Head_dict[(1, 1)] = dgl.sum_nodes(g, 'r', 'weight_0')

			# merge to 1 channel
			head_output = Tensor_fiber(tensor=Head_dict)
			predict_tensor = self.linear_head(head_output)

			# transform back to xyz
			r_vector = torch.einsum('ij,bcj->bci', self.Q_tensor, predict_tensor.get_component((1, 1))).squeeze(1)
			trans_b = torch.einsum('ij,bcj->bci', self.Q, predict_tensor.get_component((0, 1))).squeeze(1)
			trans_a = torch.einsum('ij,bcj->bci', self.Q, predict_tensor.get_component((1, 0))).squeeze(1)

		# compute r, t
		rot_mat = project_r(r_vector)
		t = m2 + trans_b - (rot_mat @ m1.unsqueeze(-1)).squeeze(-1) - (rot_mat @ trans_a.unsqueeze(-1)).squeeze(-1)
		return rot_mat, t, r_vector