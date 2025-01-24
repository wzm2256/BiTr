import copy
import pdb

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import dgl
import utils
import einops

class Nearest_neighbor:
	'''
	Find the nearest (Euclidean) neighbor in dst for each point in src
	Input:
		src: Nxm array of points
		dst: Nxm array of points
	Output:
		distances: Euclidean distances of the nearest neighbor
		indices: dst indices of the nearest neighbor
	'''

	def __init__(self, dst):
		self.neigh = NearestNeighbors(n_neighbors=1)
		self.neigh.fit(dst)
		self.dst_size = dst.shape[0]

	def __call__(self, src, threshold_type=None, threshold=0.1):
		distances, indices = self.neigh.kneighbors(src, return_distance=True)
		if threshold_type == 'm' and threshold < 1.0:
			# pdb.set_trace()
			actual_threshold = int(threshold * min(self.dst_size, src.shape[0]))
			index_src = np.argpartition(distances.ravel(), int(actual_threshold))[:int(actual_threshold)]
		elif threshold_type is None or (threshold_type == 'm' and threshold == distances.ravel().shape[0]):
			index_src = list(range(src.shape[0]))
		elif threshold_type == 'm':
			index_src = np.argpartition(distances.ravel(), int(threshold))[:int(threshold)]
		elif threshold_type == 'h':
			index_src = np.where(distances.ravel() < threshold)[0]
		else:
			raise NotImplementedError

		index_dst = indices.ravel()[index_src]
		distance = distances.ravel()[index_src]
		return distance, index_src, index_dst


def paired_align(x, y):
	# align y to x
	# H = np.dot(np.transpose(y - y.mean(axis=0)), x - x.mean(axis=0))
	# U, S, Vt = np.linalg.svd(H)
	# R = np.dot(U, Vt)
	# if np.linalg.det(R) < 0:
	# 	Vt[2, :] *= -1
	# 	R = np.dot(U, Vt)
	# t = np.mean(np.array(x - y.dot(R)), axis=0, keepdims=True)

	# align x to y
	H = np.dot(np.transpose(y - y.mean(axis=0)), x - x.mean(axis=0))
	U, S, Vt = np.linalg.svd(H)
	R = np.dot(U, Vt)
	if np.linalg.det(R) < 0:
		Vt[2, :] *= -1
		R = np.dot(U, Vt)
	# pdb.set_trace()
	# t = np.mean(np.array(y - x.dot(R.T)), axis=0, keepdims=True)
	t = np.mean(y - x.dot(R.T), axis=0, keepdims=True)

	return R, t


def icp(A, B, init_A=None, init_t=None,  max_iterations=500, tolerance=0.001,
        threshold_type='h',threshold=1.0, verbose=False):
	'''
	The Iterative Closest Point method: finds best-fit transform that maps points A on to points B.
	Transformation can be rigid or non-rigid.
	Input:
		A: Nx3 numpy array of source mD points
		B: Nx3 numpy array of reference mD point
	'''

	# initialization
	if init_A is None or init_t is None:
		R = np.eye(3)
		t = np.zeros((1, 3))
	else:
		R, t = init_A, init_t


	nn = Nearest_neighbor(B)
	for i in range(max_iterations):
		old_para = copy.deepcopy([R, t])
		Transformed_src = A.dot(R.T) + t

		distances, index_src, index_dst = nn(Transformed_src, threshold_type=threshold_type, threshold=threshold)

		# pdb.set_trace()
		if distances.shape[0] <= 5:
			return R, t
		R, t = paired_align(A[index_src, :], B[index_dst, :])
		para = [R, t]
		Diff_i = np.array([np.sum(np.abs(old_para[i] - para[i])) for i in range(len(para))]).sum()

		if verbose == True:
			print('Update difference {} at iteration {}'.format(Diff_i, i))
		if Diff_i < tolerance:
			break

	return R, t


def icp_batch_graph(g1, g2, r_gt, t_gt, r=None, t=None):
	# pdb.set_trace()
	r_iso_list = []
	t_iso_list = []
	g1_list = dgl.unbatch(g1)
	g2_list = dgl.unbatch(g2)
	if r is None:
		r = einops.repeat(torch.eye(3), 'h w -> c h w', c=len(g2_list))
		t = einops.repeat(torch.zeros(3), 'h -> c h', c=len(g2_list))

	r_icp_list = []
	t_icp_list = []
	for batch_i in range(len(g2_list)):
		r_icp, t_icp = icp(g1_list[batch_i].ndata['pos'].cpu().detach().numpy(),
		                       g2_list[batch_i].ndata['pos'].cpu().detach().numpy(),
		                       init_A=r.cpu().detach().numpy()[batch_i],
		                       init_t=np.expand_dims(t.cpu().detach().numpy()[batch_i], 0),
		                       # threshold_type='h',threshold=0.2)
								threshold_type = 'm', threshold = 0.8)
								# threshold_type = 'm', threshold = 0.6)
		# pdb.set_trace()
		r_isotropic_icp, t_isotropic_icp, _ = utils.compute_metrics(torch.asarray(r_icp).to('cuda').unsqueeze(0),
		                                                 torch.asarray(t_icp).to('cuda'), r_gt[batch_i].unsqueeze(0),
		                                                t_gt[batch_i].unsqueeze(0))
		r_iso_list.append(r_isotropic_icp)
		t_iso_list.append(t_isotropic_icp)
		r_icp_list.append(r_icp)
		t_icp_list.append(t_icp)

	r_isotropic = torch.asarray(r_iso_list)
	t_isotropic = torch.asarray(t_iso_list)
	r_icp_all = torch.asarray(r_icp_list)
	t_icp_all = torch.asarray(t_icp_list).squeeze(0)
	# pdb.set_trace()

	return r_isotropic, t_isotropic, r_icp_all, t_icp_all


# import open3d as o3d
# def icp_open3d(source, target, distance):
#     def npy2pcd(npy):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(npy)
#         return pcd
#
#     max_correspondence_distance = distance # 0.5 in RPM-Net
#     init = np.eye(4, dtype=np.float32)
#     estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     PC_source = npy2pcd(source)
#     Ref_source = npy2pcd(target)
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source=PC_source,
#         target=Ref_source,
#         init=init,
#         max_correspondence_distance=max_correspondence_distance,
#         estimation_method=estimation_method
#     )
#
#     transformation = reg_p2p.transformation
#     estimate = copy.deepcopy(PC_source)
#     estimate.transform(transformation)
#     R, t = transformation[:3, :3], transformation[:3, 3]
#     return R, t, np.asarray(estimate.points)