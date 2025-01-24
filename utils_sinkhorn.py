import argparse
import logging
import pdb
import einops
import numpy as np
import torch
import torch.nn as nn
import dgl
import utils
_EPS = 1e-5
def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
	""" Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

	Args:
		log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
		n_iters (int): Number of normalization iterations
		slack (bool): Whether to include slack row and column
		eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

	Returns:
		log(perm_matrix): Doubly stochastic matrix (B, J, K)

	Modified from original source taken from:
		Learning Latent Permutations with Gumbel-Sinkhorn Networks
		https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
	"""

	# Sinkhorn iterations
	prev_alpha = None
	if slack:
		zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
		log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

		log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

		for i in range(n_iters):
			# Row normalization
			log_alpha_padded = torch.cat((
					log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
					log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
				dim=1)

			# Column normalization
			log_alpha_padded = torch.cat((
					log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
					log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
				dim=2)

			if eps > 0:
				if prev_alpha is not None:
					abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
					if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
						print(f'Sinkhorn {i}')
						break
				prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

		log_alpha = log_alpha_padded[:, :-1, :-1]
	else:
		for i in range(n_iters):
			# Row normalization (i.e. each row sum to 1)
			log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

			# Column normalization (i.e. each column sum to 1)
			log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

			if eps > 0:
				if prev_alpha is not None:
					abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
					if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
						break
				prev_alpha = torch.exp(log_alpha).clone()

	return log_alpha




def compute_affinity(beta, feat_distance, alpha=0.5):
	"""Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
	if isinstance(alpha, float):
		hybrid_affinity = -beta * (feat_distance - alpha)
	else:
		hybrid_affinity = -beta * (feat_distance - alpha[:, None, None])
	return hybrid_affinity

def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
	"""Compute rigid transforms between two point sets

	Args:
		a (torch.Tensor): (B, M, 3) points
		b (torch.Tensor): (B, N, 3) points
		weights (torch.Tensor): (B, M)

	Returns:
		Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
	"""

	weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
	centroid_a = torch.sum(a * weights_normalized, dim=1)
	centroid_b = torch.sum(b * weights_normalized, dim=1)
	a_centered = a - centroid_a[:, None, :]
	b_centered = b - centroid_b[:, None, :]
	cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

	# Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
	# and choose based on determinant to avoid flips
	u, s, v = torch.svd(cov, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(rot_mat) > 0)

	# Compute translation (uncenter centroid)
	translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

	transform = torch.cat((rot_mat, translation), dim=2)
	return transform

def transform_se3(g, a, normals=None):
	""" Applies the SE3 transform

	Args:
		g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
		a: Points to be transformed (N, 3) or (B, N, 3)
		normals: (Optional). If provided, normals will be transformed

	Returns:
		transformed points of size (N, 3) or (B, N, 3)

	"""
	R = g[..., :3, :3]  # (B, 3, 3)
	p = g[..., :3, 3]  # (B, 3)

	if len(g.size()) == len(a.size()):
		b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
	else:
		raise NotImplementedError
		# b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

	if normals is not None:
		rotated_normals = normals @ R.transpose(-1, -2)
		return b, rotated_normals

	else:
		return b



def OT(src, ref, r, t, eps=100., threshold=0.2, iter=10, iter_sk=10):

	# r_tmp = r.copy()
	# t_tmp = t.copy()
	src_tmp = (src @ r.T + t).unsqueeze(0)

	# pdb.set_trace()
	# transform_old = None
	transform_old = torch.zeros((1,3,4), device=src_tmp.device)
	for i in range(iter):
		# pdb.set_trace()
		D = ((src_tmp.squeeze(0).unsqueeze(1) - ref.unsqueeze(0)) ** 2).sum(-1).unsqueeze(0)

		# pdb.set_trace()
		affinity = compute_affinity(eps, D, alpha=threshold ** 2)

		# Compute weighted coordinates
		log_perm_matrix = sinkhorn(affinity, n_iters=iter_sk, slack=True, eps=1e-5)
		# log_perm_matrix = sinkhorn(affinity, n_iters=iter_sk, slack=True)
		perm_matrix = torch.exp(log_perm_matrix)
		weighted_ref = perm_matrix @ ref.unsqueeze(0) / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

		# Compute transform and transform points
		transform = compute_rigid_transform(src.unsqueeze(0), weighted_ref, weights=torch.sum(perm_matrix, dim=2))
		# pdb.set_trace()
		# if transform_old
		src_tmp = transform_se3(transform.detach(), src.unsqueeze(0))
		if (transform_old - transform).norm() < 1e-6:
			print(f'Run {i} steps')
			break
		else:
			transform_old = transform.clone()


	R = transform[..., :3, :3]  # (B, 3, 3)
	p = transform[..., :3, 3]  # (B, 3)
	# pdb.set_trace()
	return R, p


def OT_compute(g1, g2, r_gt, t_gt, r=None, t=None):
	# pdb.set_trace()
	r_iso_list = []
	t_iso_list = []
	g1_list = dgl.unbatch(g1)
	g2_list = dgl.unbatch(g2)
	if r is None:
		# raise NotImplementedError
		r = einops.repeat(torch.eye(3), 'h w -> c h w', c=len(g2_list))
		t = einops.repeat(torch.zeros(3), 'h -> c h', c=len(g2_list))

	r_icp_list = []
	t_icp_list = []
	for batch_i in range(len(g2_list)):
		r_icp, t_icp = OT(g1_list[batch_i].ndata['pos'].detach().to('cuda'),
							   g2_list[batch_i].ndata['pos'].detach().to('cuda'),
							   r=r.detach()[batch_i].to('cuda'),
							   t=t.detach()[batch_i].to('cuda').unsqueeze(0),
							   # threshold_type='h',threshold=0.2)
								eps = 5000., threshold = 0.7, iter = 100, iter_sk = 200)
								# eps = 5000., threshold = 0.6, iter = 300, iter_sk = 200)
								# threshold_type = 'm', threshold = 0.6)
		# pdb.set_trace()
		r_icp = r_icp.squeeze(0).cpu().numpy()
		t_icp = t_icp.cpu().numpy()
		r_isotropic_icp, t_isotropic_icp = utils.compute_metrics(torch.asarray(r_icp).to('cuda').unsqueeze(0),
														 torch.asarray(t_icp).to('cuda'), r_gt[batch_i].unsqueeze(0),
														t_gt[batch_i].unsqueeze(0))
		r_iso_list.append(r_isotropic_icp)
		t_iso_list.append(t_isotropic_icp)
		r_icp_list.append(r_icp)
		t_icp_list.append(t_icp)
	# pdb.set_trace()
	r_isotropic = torch.asarray(r_iso_list)
	t_isotropic = torch.asarray(t_iso_list)
	# pdb.set_trace()
	r_icp_all = torch.asarray(r_icp_list)
	t_icp_all = torch.asarray(t_icp_list).squeeze(0)
	return r_isotropic, t_isotropic, r_icp_all, t_icp_all

