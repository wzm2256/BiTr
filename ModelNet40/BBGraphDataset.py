import pdb
import numpy as np
import dgl
import torch
from .BasePCGraphData import BasePCGraph
from .transforms.functional import random_sample_points
from .utils.common import load_pickle
from typing import Dict
import os.path as osp

class BBPair(torch.utils.data.Dataset):
    # fmt: off
    def __init__(
        self,
        dataset_root: str,
        subset: str,
        category: str = '',
        return_normals: bool = False,
        keep_ratio=-1,
    ):
        super(BBPair, self).__init__()

        assert subset in ['train', 'val']
        self.dataset_root = dataset_root
        self.subset = subset
        self.return_normals = return_normals
        data_list = load_pickle(osp.join(dataset_root, f'{category}_{subset}.pickle'))
        self.data_list = data_list
        self.keep_ratio = keep_ratio

    def __getitem__(self, index):
        data_dict: Dict = self.data_list[index]

        if self.keep_ratio > 0:
            if self.return_normals:
                ref_points, ref_noramls = random_sample_points(data_dict['reference'].astype(np.float32),
                                                               int(data_dict['reference'].shape[0] * self.keep_ratio),
                                                               normals=data_dict['reference_norm'].astype(np.float32))
                src_points, src_normals = random_sample_points(data_dict['source'].astype(np.float32),
                                                               int(data_dict['source'].shape[0] * self.keep_ratio),
                                                               normals=data_dict['source_norm'].astype(np.float32))
            else:
                ref_points = random_sample_points(data_dict['reference'].astype(np.float32),
                                            int(data_dict['reference'].shape[0] * self.keep_ratio),
                                            normals=None)
                src_points = random_sample_points(data_dict['source'].astype(np.float32),
                                            int(data_dict['source'].shape[0] * self.keep_ratio),
                                            normals=None)

        else:
            ref_points = data_dict['reference'].astype(np.float32)
            src_points = data_dict['source'].astype(np.float32)
            if self.return_normals:
                ref_noramls = data_dict['reference_norm'].astype(np.float32)
                src_normals = data_dict['source_norm'].astype(np.float32)

        new_data_dict = {
            'ref_points': ref_points,
            'src_points': src_points,
            'transform': np.eye(4).astype(np.float32),
            'index': int(index),
        }

        if self.return_normals:
            new_data_dict['ref_normals'] = ref_noramls
            new_data_dict['src_normals'] = src_normals

        return new_data_dict

    def __len__(self):
        return len(self.data_list)


class BBGraph(BasePCGraph):
	def __init__(self, dataset_root='Data/BB/', subset='train', class_indices='', return_normals=True,
	             k_neighbor=5, keep_ratio=-1, **kwargs):
		self.dataset_root = dataset_root

		if subset == 'train':
			subset_ = subset
		elif subset == 'test':
			subset_ = 'val'
		else:
			raise NotImplementedError

		bb = BBPair(dataset_root, subset=subset_, category=class_indices, return_normals=return_normals, keep_ratio=keep_ratio)
		param_str = f'{class_indices}_{subset}_{return_normals}_{k_neighbor}'
		super().__init__(pc_dataset=bb, dataset_root=dataset_root, dataset_name='bb', param_str=param_str,
		                 k_neighbor=k_neighbor)

if __name__ == '__main__':

###########################################

	import args
	import time
	FLAGS = args.get_flags()

	train_dataset = BBGraph(
		subset='train',
		num_points=FLAGS.num_points,
		noise_magnitude=FLAGS.noise_magnitude,
		keep_ratio=FLAGS.keep_ratio,
		class_indices=FLAGS.class_indices,
		return_normals=FLAGS.return_normals,
		rotation_magnitude=FLAGS.rotation_magnitude,
		translation_magnitude=FLAGS.translation_magnitude,
		k_neighbor=FLAGS.k_neighbor,
		get_color=FLAGS.get_color,
		)

	# pdb.set_trace()
	train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False,
	                                               drop_last=False, num_workers=FLAGS.num_workers, pin_memory=True)

	t1 = time.time()
	pdb.set_trace()
	for i, data in enumerate(train_loader):
		all_points = torch.cat([data[0].ndata['pos'], data[0].ndata['pos']], 0)
		all_points.mean(0)
		# print('--')
		pdb.set_trace()
		pass

	t2 = time.time()

	print(f'time: {t2-t1}')