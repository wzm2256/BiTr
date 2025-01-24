import os
import pdb
import numpy as np
# import dgl
import torch
# from dgl.data import DGLDataset
# from .dataset import MugPair
# import einops
# from .modelnet_utils import convert_to_graph_mug, get_basis_new
# from .modelnet_utils import convert_to_graph
from .PCGraphData import BasePCGraph
import pickle

from .transforms.functional import (
	normalize_points,
	random_jitter_points,
	random_seperate_point_cloud,
	random_shuffle_points,
	random_sample_points,
	random_crop_point_cloud_with_plane,
	random_sample_viewpoint,
	random_crop_point_cloud_with_point,
	normalize_points_two
)

class MugPair(torch.utils.data.Dataset):
	def __init__(self, dataset_root: str='Data/manipulate', subset: str = 'train', get_color=False, noise_magnitude=None, task='mug',
	):
		super(MugPair, self).__init__()
		# assert task == 'mug'
		# self.num_points = num_points
		if task == 'mug':
			self.scale = 30.
			if subset == 'test':
				self.class_num = [6, 9, 10]
			elif subset == 'train':
				self.class_num = [0,1,4]
			else:
				raise NotImplementedError
		elif task == 'bowl':
			self.scale = 30.
			if subset == 'test':
				self.class_num = [6, 7, 8]
			elif subset == 'train':
				self.class_num = [1,2,5]
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

		# self.dataset_root = dataset_root
		self.get_color = get_color
		self.noise_magnitude = noise_magnitude
		self.data_list = []
		for i in self.class_num:
			with open(os.path.join(dataset_root, task, f'{task}_{i}.pk'), 'rb') as f:
				data_class = pickle.load(f)
				self.data_list += data_class


	def __getitem__(self, index):
		data_dict = self.data_list[index]

		if self.noise_magnitude is not None:
			ref_points = random_jitter_points(data_dict['Ref'] / self.scale, scale=0.01, noise_magnitude=self.noise_magnitude)
			src_points = random_jitter_points(data_dict['Src'] / self.scale, scale=0.01, noise_magnitude=self.noise_magnitude)
		else:
			ref_points = data_dict['Ref'] / self.scale
			src_points = data_dict['Src'] / self.scale


		transform = np.eye(4, dtype=np.float32)
		transform[:3, :3] = data_dict['R']
		transform[:3, 3] = data_dict['t'] / self.scale

		data_item={}
		data_item['transform'] = transform.astype(np.float32)
		data_item['ref_points'] = ref_points.astype(np.float32)
		data_item['src_points'] = src_points.astype(np.float32)
		data_item['index'] = index

		if self.get_color:
			data_item['Ref_color'] = np.clip(data_dict['Ref_c'] * 0.5 + 0.5, 0, 1).astype(np.float32)
			data_item['Src_color'] = np.clip(data_dict['Src_c'] * 0.5 + 0.5, 0, 1).astype(np.float32)

		return data_item

	def __len__(self):
		return len(self.data_list)


class MugGraph(BasePCGraph):
	def __init__(self, dataset_root='Data/manipulate', subset='train', get_color=False, class_indices='mug', max_deg=2,
	             k_neighbor=7, on_the_fly='simple', noise_magnitude=None, **kwargs):

		manipulate = MugPair(task=class_indices, dataset_root=dataset_root, subset=subset, get_color=get_color,
		                          noise_magnitude=noise_magnitude)

		param_str = f'{class_indices}_{subset}_{get_color}'
		super().__init__(pc_dataset=manipulate, dataset_root=dataset_root, dataset_name='MugGraph',
						 param_str=param_str, max_deg=max_deg, on_the_fly=on_the_fly, k_neighbor=k_neighbor)


if __name__ == '__main__':
	# dataset = ModelnetGraph(force_reload=True)
	import utils
	dataset = MugPair(subset='test', get_color=True, task='bowl', noise_magnitude=None)

	# pdb.set_trace()
	for i in range(len(dataset)):
		pc = dataset[i]
		pc1 = pc['ref_points']
		pc2 = pc['src_points']

		#
		color_src = pc['Src_color']
		color_ref = pc['Ref_color']

		transform = pc['transform']
		r = transform[0:3, 0:3]
		t = transform[0:3, 3:]
		pc2_pre = pc2 @ r.T + t.T
		pdb.set_trace()
		utils.visualize_3d_3_color([pc2, pc1, pc2_pre], [color_src, color_ref, color_src])

		# utils.vis_PC([pc2, pc1, pc2_pre])

		# data_item['src_points'] = src_points
	# dataset = MugGraph(
	# 	dataset_root='Data/demos_mug',
	# 	subset='test',
	# 	get_color=False,
	# 	minimal_edge_len=0.03,
	# 	max_deg=2,
	# 	k_neighbor=2,
	# 	force_reload=False
	# )

	# pdb.set_trace()
	# train_loader = dgl.dataloading.GraphDataLoader(dataset, batch_size=1, shuffle=True,
	#                                                drop_last=False, num_workers=0)

	# for _ in range(5):
	# 	for g_ref, g_src, transform in train_loader:
	#
	# 		pc1 = g_ref.ndata['pos'].numpy()
	# 		pc2 = g_src.ndata['pos'].numpy()
	# 		r = transform[0, 0:3,0:3].numpy()
	# 		t = transform[0, 0:3, 3:].numpy()
	# 		pc2_pre = pc2 @ r.T + t.T
	# 		import utils
	# 		utils.vis_PC([pc2, pc1, pc2_pre])
