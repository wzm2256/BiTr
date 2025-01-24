import pdb
import dgl
from .ModelNetPC import ModelNetPairDataset
from .data_utils import random_crop_with_plane_graph
from .BasePCGraphData import BasePCGraph

class ModelnetGraph(BasePCGraph):
	def __init__(self, subset='train',  # val, test
	             num_points=1024, noise_magnitude=0.02, keep_ratio=0.7, class_indices=None,
	             return_normals=True,
	             rotation_magnitude=45, translation_magnitude=0.5,
	             k_neighbor=5,
	             **kwargs):

		self.pre_compute = False
		self.keep_ratio = keep_ratio

		if class_indices is None:
			data_class_indices = [0]
		else:
			data_class_indices = class_indices

		dataset_root = 'Data/ShapeNet'

		self.modenet40 = ModelNetPairDataset(dataset_root=dataset_root, subset=subset, num_points=num_points,
		            noise_magnitude=noise_magnitude, keep_ratio=keep_ratio, class_indices=data_class_indices,
		            return_normals=return_normals, rotation_magnitude=rotation_magnitude,
		            translation_magnitude=translation_magnitude)

		param_str = f'{class_indices}_{subset}_{num_points}_{noise_magnitude}_{keep_ratio}_{return_normals}_{k_neighbor}'

		super().__init__(pc_dataset=self.modenet40, dataset_root=dataset_root, dataset_name='ModelNet40', param_str=param_str,
		                 k_neighbor=k_neighbor)

	def __getitem__(self, i):
		if not self.pre_compute:
			graph_ref, graph_src, transform = super().__getitem__(i)
			return graph_ref, graph_src, transform
		else:
			g_ref, g_src, transform = super().__getitem__(i)
			if self.keep_ratio is not None:
				g_ref = random_crop_with_plane_graph(g_ref, p_normal=None, keep_ratio=self.keep_ratio)
				g_src = random_crop_with_plane_graph(g_src, p_normal=None, keep_ratio=self.keep_ratio)
			return g_ref, g_src, transform

	def __len__(self):
		return len(self.modenet40)


if __name__ == '__main__':

###########################################
	import utils
	import args
	import time
	FLAGS = args.get_flags()

	train_dataset = ModelnetGraph(
		dataset_root='Data/ShapeNet',
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

	train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False,
	                                               drop_last=False, num_workers=FLAGS.num_workers, pin_memory=True)


	for i, data in enumerate(train_loader):
		source_PC = data[1].ndata['pos']
		Ref_PC = data[0].ndata['pos']
		utils.vis_PC([source_PC, Ref_PC], point_size=5.)
