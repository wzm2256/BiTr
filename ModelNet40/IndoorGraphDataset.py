import os
import pdb
import numpy as np
import torch
from .BasePCGraphData import BasePCGraph
from .utils.pointcloud import random_sample_transform, apply_transform, inverse_transform
from .transforms.functional import random_sample_points


class MatchPair(torch.utils.data.Dataset):
	"""
	Load subsampled coordinates, relative rotation and translation
	Output(torch.Tensor):
		src_pcd:        [N,3]
		tgt_pcd:        [M,3]
		rot:            [3,3]
		trans:          [3,1]
	"""

	def __init__(self,
        subset: str,
        category: str = '',
        return_normals: bool = False,
        keep_ratio=-1,
        noise_magnitude=0.,
        dataset_root='Data/Match',
		rotation_magnitude: float = 0.,
		translation_magnitude: float = 0.,
		):
		super(MatchPair, self).__init__()

		self.category = category
		self.base_dir = dataset_root
		self.max_points = 20000
		self.keep_ratio = keep_ratio

		if subset == 'train':
			self.data_augmentation = True
			All_list_file = open(os.path.join(self.base_dir, 'train_adj.txt'))
		else:
			self.data_augmentation = False
			All_list_file = open(os.path.join(self.base_dir, 'test_adj.txt'))

			self.keep_ratio = -1.0
			self.max_points = 200000000

		if category == 'All':
			self.All_list = [l.strip().split('\t') for l in All_list_file.readlines()]
		else:
			self.All_list = [l.strip().split('\t') for l in All_list_file.readlines() if self.category in l]

		self.augment_noise = noise_magnitude
		self.return_normals = return_normals

		self.rotation_magnitude = rotation_magnitude
		self.translation_magnitude = translation_magnitude

	def __len__(self):
		return len(self.All_list)

	def __getitem__(self, index):

		src_path = os.path.join(self.base_dir, self.All_list[index][0])
		tgt_path = os.path.join(self.base_dir, self.All_list[index][1])

		src_pcd = torch.load(src_path).astype(np.float32)
		tgt_pcd = torch.load(tgt_path).astype(np.float32)

		if (src_pcd.shape[0] > self.max_points):
			idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
			src_pcd = src_pcd[idx]
		if (tgt_pcd.shape[0] > self.max_points):
			idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
			tgt_pcd = tgt_pcd[idx]


		src_trans_path = src_path.split('.')[0] + '.info.txt'
		tgt_trans_path = tgt_path.split('.')[0] + '.info.txt'
		if os.path.isfile(src_trans_path):
			src_trans = np.loadtxt(src_trans_path, skiprows=1, usecols=None).astype(np.float32)
			tgt_trans = np.loadtxt(tgt_trans_path, skiprows=1, usecols=None).astype(np.float32)

			tgt_move = tgt_pcd @ tgt_trans[:3, :3].T + tgt_trans[:3, 3:].T
			src_move = src_pcd @ src_trans[:3, :3].T + src_trans[:3, 3:].T
		else:
			tgt_move = tgt_pcd
			src_move = src_pcd

		if self.return_normals:
			src_norm_path = src_path.split('.')[0] + '_norm.pth'
			tgt_norm_path = tgt_path.split('.')[0] + '_norm.pth'


			tgt_pcd_norm = torch.load(tgt_norm_path).astype(np.float32)
			src_pcd_norm = torch.load(src_norm_path).astype(np.float32)

			# pdb.set_trace()
			if os.path.isfile(src_trans_path):
				tgt_pcd_norm = tgt_pcd_norm @ tgt_trans[:3, :3].T
				src_pcd_norm = src_pcd_norm @ src_trans[:3, :3].T


		else:
			tgt_pcd_norm = None
			src_pcd_norm = None

		transform_src = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
		inv_transform_src = inverse_transform(transform_src)
		src_move, src_pcd_norm = apply_transform(src_move, inv_transform_src, normals=src_pcd_norm)

		transform_ref = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
		tgt_move, tgt_pcd_norm = apply_transform(tgt_move, transform_ref, normals=tgt_pcd_norm)


		if self.keep_ratio > 0:
			keep_ratio1 = np.random.rand() * (1 - self.keep_ratio) + self.keep_ratio
			keep_ratio2 = np.random.rand() * (1 - self.keep_ratio) + self.keep_ratio

			if self.return_normals:
				ref_points, ref_noramls = random_sample_points(tgt_move,
				                                               int(tgt_pcd.shape[0] * keep_ratio1),
				                                               normals=tgt_pcd_norm)
				src_points, src_normals = random_sample_points(src_move,
				                                               int(src_pcd.shape[0] * keep_ratio2),
				                                               normals=src_pcd_norm)
			else:
				ref_points = random_sample_points(tgt_move,
				                                  int(tgt_pcd.shape[0] * keep_ratio1),
				                                  normals=None)
				src_points = random_sample_points(src_move,
				                                  int(src_move.shape[0] * keep_ratio2),
				                                  normals=None)
		else:
			ref_points = tgt_move
			src_points = src_move
			if self.return_normals:
				ref_noramls = tgt_pcd_norm
				src_normals = src_pcd_norm


		if self.data_augmentation:
			src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augment_noise
			ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augment_noise


		# pdb.set_trace()
		new_data_dict = {
			'ref_points': ref_points,
			'src_points': src_points,
			'transform': (transform_ref @ transform_src).astype(np.float32),
			'index': int(index),
		}

		if self.return_normals:
			new_data_dict['ref_normals'] = ref_noramls
			new_data_dict['src_normals'] = src_normals

		return new_data_dict



class MatchGraph(BasePCGraph):
	def __init__(self, dataset_root='Data/Match01', subset='train', class_indices='', return_normals=True,
	             k_neighbor=5, keep_ratio=-1, noise_magnitude=0., rotation_magnitude=0.,
	             translation_magnitude=0., **kwargs):
		self.dataset_root = dataset_root

		if subset != 'train' and subset != 'test':
			raise NotImplementedError

		bb = MatchPair(dataset_root=dataset_root, subset=subset, category=class_indices, return_normals=return_normals,
		               noise_magnitude=noise_magnitude, keep_ratio=keep_ratio, rotation_magnitude=rotation_magnitude,
		               translation_magnitude=translation_magnitude)
		param_str = f'{class_indices}_{subset}_{return_normals}_{k_neighbor}'
		super().__init__(pc_dataset=bb, dataset_root=dataset_root, dataset_name='match', param_str=param_str,
		                 k_neighbor=k_neighbor)



if __name__ == '__main__':


	import utils
	###########################################
	Dataset = MatchPair(
	             subset='train',
	             category='7-scenes',
	             return_normals=True,
	             keep_ratio=-1,
	             noise_magnitude=0.005,
				dataset_root = 'Data/Match01',
	             )

	print(len(Dataset))

	minsize = 10000
	maxsize = 0
	for d in Dataset:
		size1 = d['ref_points'].shape[0]
		size2 = d['src_points'].shape[0]
		if min(size1, size2) < minsize:
			minsize = min(size1, size2)
		if min(size1, size2) > maxsize:
			maxsize = min(size1, size2)

	print(f'minsize {minsize}  maxsize {maxsize}')
