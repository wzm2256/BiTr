import numpy as np
import einops
try:
	from escnn.kernels.harmonic_polynomial_r3 import HarmonicPolynomialR3Generator
	import escnn.group as g
except:
	print('Warning: Can not import escnn. Load processed data only.')

from dgl import backend as F
import dgl
import torch


def knn_graph_with_min(x, k=1, min_dist=0.0):
	if F.ndim(x) == 2:
		x = F.unsqueeze(x, 0)
	n_samples, n_points, _ = F.shape(x)

	ctx = F.context(x)
	# dist = pairwise_squared_distance(x)

	dist = torch.cdist(x, x, p=2)
	edge_select = dist <= min_dist
	dist[edge_select] = torch.inf

	k_indices = F.astype(F.argtopk(dist, k, 2, descending=False), F.int64)
	assert torch.sum(F.topk(dist, k, 2, descending=False)) < torch.inf, 'Not enough points'
	offset = F.arange(0, n_samples, ctx=ctx) * n_points
	offset = F.unsqueeze(offset, 1)
	src = F.reshape(k_indices, (n_samples, n_points * k))
	src = F.unsqueeze(src, 0) + offset
	dst = F.repeat(F.arange(0, n_points, ctx=ctx), k, dim=0)
	dst = F.unsqueeze(dst, 0) + offset
	return dgl.convert.graph((F.reshape(src, (-1,)), F.reshape(dst, (-1,))))


def radius_graph_with_min(
	x,
	r,
	r_min=0.0,
	p=2,
	self_loop=False,
	compute_mode="donot_use_mm_for_euclid_dist",
	get_distances=False,
):
	distances = torch.cdist(x, x, p=p, compute_mode=compute_mode)

	if not self_loop:
		distances.fill_diagonal_(r + 1)

	edge_select = torch.logical_and(distances <= r, distances >= r_min)
	edges = torch.nonzero(edge_select, as_tuple=True)

	g = dgl.convert.graph(edges, num_nodes=x.shape[0], device=x.device)

	if get_distances:
		distances = distances[edges].unsqueeze(-1)

		return g, distances

	return g



def random_crop_with_plane_graph(graph, p_normal=None, keep_ratio=0.7, device='cpu'):
	r"""Random crop a point cloud with a plane and keep num_samples points."""
	num_remove_samples = int(np.floor(graph.num_nodes() * (1 - keep_ratio) + 0.5))
	if p_normal is None:
		p_normal = torch.randn((1, 3), device=device)  # (3,)
	distances = (graph.ndata['pos'] * p_normal).sum(-1)
	sel_indices = torch.topk(distances, num_remove_samples)[1] # select the largest K points

	G_new = dgl.remove_nodes(graph, sel_indices)
	return G_new


def get_basis_new(max_degree, G):
	device = G.device
	G.edata['d'] = G.ndata['pos'][G.edges()[1]] - G.ndata['pos'][G.edges()[0]]
	harmonics_generator = HarmonicPolynomialR3Generator(2*max_degree).to(device)
	sphere1 = torch.nn.functional.normalize(G.edata['d'], dim=1)
	Y_escnn1 = harmonics_generator(sphere1)
	r1 = g.so3_group(3)
	basis_new = {}
	for in1 in range(max_degree + 1):
		for out1 in range(max_degree + 1):
			size_out = 2 * out1 + 1
			escnn_Q = r1.irreps()[in1].tensor(r1.irreps()[out1]).change_of_basis
			#####################################################
			K_Js = []
			total_length = 0
			for J1 in range(abs(in1-out1), in1+out1+1):
				block_size = 2 * J1 + 1
				Q_J_escnn = torch.tensor(escnn_Q[:, total_length: total_length + block_size], dtype=torch.float, device=device)
				total_length += block_size
				Y1 = Y_escnn1[:, J1 ** 2: (J1 + 1) ** 2]
				K_J = torch.matmul(Q_J_escnn, Y1.T)
				K_Js.append(K_J)
			new = einops.rearrange(K_Js, 'f (i o) p -> p 1 o 1 i f', o=size_out)
			basis_new[f'basis_(({in1}, 0),({out1}, 0))'] = new

	for k, i in basis_new.items():
		G.edata[k] = i


	# d1 = torch.sqrt(torch.sum(G.edata['d'] ** 2, -1, keepdim=True))
	d1 = torch.linalg.vector_norm(G.edata['d'], dim=-1, keepdim=True)
	# pdb.set_trace()
	#####
	# d2 = torch.zeros_like(d1)
	# G.edata['r'] = torch.cat([d1, d2], -1)
	#####
	G.edata['r'] = d1
	#####
	return


def convert_to_graph(pc, max_deg=2, radius=0.05, minimal_edge_len=0.0, k_neighbor=5, use_knn=False, type='simple', return_normals=False,):

	device='cpu'

	if not use_knn:
		raise NotImplementedError

	graph_ref = dgl.knn_graph(torch.tensor(pc['ref_points'], device=device), k_neighbor, exclude_self=True)
	graph_src = dgl.knn_graph(torch.tensor(pc['src_points'], device=device), k_neighbor, exclude_self=True)

	graph_ref.ndata['pos'] = torch.tensor(pc['ref_points'], dtype=torch.float, device=device)
	graph_src.ndata['pos'] = torch.tensor(pc['src_points'], dtype=torch.float, device=device)

	if 'src_normals' in pc.keys():
		graph_ref.ndata['norm_vec'] = torch.tensor(pc['ref_normals'], dtype=torch.float, device=device)
		graph_src.ndata['norm_vec'] = torch.tensor(pc['src_normals'], dtype=torch.float, device=device)

	if 'Src_color' in pc.keys():
		graph_ref.ndata['c'] = torch.tensor(pc['Ref_color'], dtype=torch.float, device=device)
		graph_src.ndata['c'] = torch.tensor(pc['Src_color'], dtype=torch.float, device=device)
	else:
		graph_ref.ndata['c'] = torch.ones((pc['ref_points'].shape[0], 1), dtype=torch.float, device=device)
		graph_src.ndata['c'] = torch.ones((pc['src_points'].shape[0], 1), dtype=torch.float, device=device)

	if type != 'simple':
		raise NotImplementedError

	# if type == 'basis':
	# 	g_batch = dgl.batch([graph_ref, graph_src])
	# 	get_basis_new(max_deg, g_batch)
	# 	graph_ref, graph_src = dgl.unbatch(g_batch)
	# elif type == 'simple':
	# 	pass
	# else:
	# 	raise NotImplementedError
	return graph_ref, graph_src

def catpc2_graph(pc1, pc2, k_neighbor=5, num_kp=1):
	'''
	Concate two key point PCs, and get the nearest neighbors graph.
	'''
	pc_cat = torch.cat([pc1, pc2], 1)
	pc_cat_batch = einops.rearrange(pc_cat, '(b p) d -> b p d', p=num_kp)
	graph = dgl.knn_graph(pc_cat_batch, k_neighbor, exclude_self=True)
	graph.ndata['pos'] = pc_cat
	return graph
