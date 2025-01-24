import os
import numpy as np
import dgl
import torch
from torch.utils.data import Dataset
from typing import Optional
from .transforms.functional import (
	normalize_points,
	random_jitter_points,
	random_seperate_point_cloud,
	random_sample_points,
)
from .BasePCGraphData import BasePCGraph

class BunnyDataset(torch.utils.data.Dataset):

	def __init__(
		self,
		dataset_root: str,
		num_points: int = 1024,
		noise_magnitude: Optional[float] = None,
		keep_ratio: Optional[float] = 0.5,
		outlier_num=-1,
	):

		super(BunnyDataset, self).__init__()

		self.dataset_root = dataset_root

		self.num_points = num_points
		self.noise_magnitude = noise_magnitude
		self.keep_ratio = keep_ratio

		self.raw = np.loadtxt(os.path.join(dataset_root, 'bunny-x.txt'))
		self.outliers = outlier_num

	def __getitem__(self, index):
		raw_points = normalize_points(self.raw)
		raw_points = random_sample_points(raw_points, self.num_points, use_random=True)

		if self.outliers > 0:
			raw_points = np.concatenate([(np.random.rand(self.outliers, 3) - 0.5) * 2, raw_points], axis=0)

		if self.keep_ratio > 0:
			ref_points, src_points = random_seperate_point_cloud(raw_points, p_normal=None, keep_ratio=self.keep_ratio, use_random_plane=True)
		else:
			ref_points = raw_points
			src_points = raw_points

		if self.noise_magnitude is not None:
			ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
			src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

		new_data_dict = {
			'ref_points': ref_points.astype(np.float32),
			'src_points': src_points.astype(np.float32),
			'transform': np.eye(4, dtype=np.float32),
			'index': int(index),
		}

		return new_data_dict

	def __len__(self):
		return 1


class BunnyGraph(BasePCGraph):
	def __init__(self,
				 dataset_root='Data/bunny',
				 num_points=1024,
				 noise_magnitude=0.02,
				 keep_ratio=0.7,
				 k_neighbor=5,
				 **kwargs):

		bunny_dataset = BunnyDataset(
			dataset_root=dataset_root,
			num_points=num_points,
			noise_magnitude=noise_magnitude,
			keep_ratio=keep_ratio,
		)

		param_str = f'{keep_ratio}_{noise_magnitude}'
		super().__init__(pc_dataset=bunny_dataset, dataset_root=dataset_root, dataset_name='Bunny',
						 param_str=param_str, k_neighbor=k_neighbor)


if __name__ == '__main__':
	import utils
	dataset = BunnyGraph(
		dataset_root='Data/bunny',
		num_points=1024,
		noise_magnitude=0.01,
		keep_ratio=0.5,
		k_neighbor=5,
		outlier_num=200
	)
	train_loader = dgl.dataloading.GraphDataLoader(dataset, batch_size=1, shuffle=True,
												   drop_last=False, num_workers=0)

	for _ in range(5):
		for g_ref, g_src, transform in train_loader:

			pc1 = g_ref.ndata['pos'].cpu().numpy()
			pc2 = g_src.ndata['pos'].cpu().numpy()
			r = transform[0, 0:3,0:3].cpu().numpy()
			t = transform[0, 0:3, 3:].cpu().numpy()
			pc2_pre = pc2 @ r.T + t.T
			# import utils
			utils.vis_PC([pc2, pc1, pc2_pre])
