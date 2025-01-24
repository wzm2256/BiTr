"""Basic DGL Dataset
"""

from __future__ import absolute_import

import abc
import hashlib
import os
import traceback
import dgl

from .data_utils import convert_to_graph, get_basis_new, random_crop_with_plane_graph
import numpy as np
from tqdm import tqdm

class MyDGLDataset(object):
	r"""
	a dgl dataset class, with a parameter pre_compute determining whether to skip the graph processing step
	"""

	def __init__(
		self,
		name='',
		url=None,
		raw_dir=None,
		save_dir=None,
		hash_key=(),
		force_reload=False,
		verbose=False,
		transform=None,
		pre_compute=True
	):
		self._name = name
		self._url = url
		self._force_reload = force_reload
		self._verbose = verbose
		self._hash_key = hash_key
		self._hash = self._get_hash()
		self._transform = transform

		self._raw_dir = raw_dir

		if save_dir is None:
			self._save_dir = self._raw_dir
		else:
			self._save_dir = save_dir

		if pre_compute:
			self._load()


	def save(self):
		r"""Overwite to realize your own logic of
		saving the processed dataset into files.

		It is recommended to use ``dgl.data.utils.save_graphs``
		to save dgl graph into files and use
		``dgl.data.utils.save_info`` to save extra
		information into files.
		"""
		pass

	def load(self):
		r"""Overwite to realize your own logic of
		loading the saved dataset from files.

		It is recommended to use ``dgl.data.utils.load_graphs``
		to load dgl graph from files and use
		``dgl.data.utils.load_info`` to load extra information
		into python dict object.
		"""
		pass

	@abc.abstractmethod
	def process(self):
		r"""Overwrite to realize your own logic of processing the input data."""
		pass

	def has_cache(self):
		r"""Overwrite to realize your own logic of
		deciding whether there exists a cached dataset.

		By default False.
		"""
		return False


	def _load(self):
		"""Entry point from __init__ to load the dataset.

		If cache exists:

		  - Load the dataset from saved dgl graph and information files.
		  - If loadin process fails, re-download and process the dataset.

		else:

		  - Download the dataset if needed.
		  - Process the dataset and build the dgl graph.
		  - Save the processed dataset into files.
		"""
		load_flag = not self._force_reload and self.has_cache()

		if load_flag:
			try:
				self.load()
				if self.verbose:
					print("Done loading data from cached files.")
			except KeyboardInterrupt:
				raise
			except:
				load_flag = False
				if self.verbose:
					print(traceback.format_exc())
					print("Loading from cache failed, re-processing.")

		if not load_flag:
			# self._download()
			self.process()
			self.save()
			if self.verbose:
				print("Done saving data into cached files.")

	def _get_hash(self):
		"""Compute the hash of the input tuple

		Example
		-------
		Assume `self._hash_key = (10, False, True)`

		>>> hash_value = self._get_hash()
		>>> hash_value
		'a770b222'
		"""
		hash_func = hashlib.sha1()
		hash_func.update(str(self._hash_key).encode("utf-8"))
		return hash_func.hexdigest()[:8]

	def _get_hash_url_suffix(self):
		"""Get the suffix based on the hash value of the url."""
		if self._url is None:
			return ""
		else:
			hash_func = hashlib.sha1()
			hash_func.update(str(self._url).encode("utf-8"))
			return "_" + hash_func.hexdigest()[:8]

	@property
	def url(self):
		r"""Get url to download the raw dataset."""
		return self._url

	@property
	def name(self):
		r"""Name of the dataset."""
		return self._name

	@property
	def raw_dir(self):
		r"""Raw file directory contains the input data folder."""
		return self._raw_dir

	@property
	def raw_path(self):
		r"""Directory contains the input data files.
		By default raw_path = os.path.join(self.raw_dir, self.name)
		"""
		return os.path.join(
			self.raw_dir, self.name + self._get_hash_url_suffix()
		)

	@property
	def save_dir(self):
		r"""Directory to save the processed dataset."""
		return self._save_dir

	@property
	def save_path(self):
		r"""Path to save the processed dataset."""
		return os.path.join(
			self.save_dir, self.name + self._get_hash_url_suffix()
		)

	@property
	def verbose(self):
		r"""Whether to print information."""
		return self._verbose

	@property
	def hash(self):
		r"""Hash value for the dataset and the setting."""
		return self._hash

	@abc.abstractmethod
	def __getitem__(self, idx):
		r"""Gets the data object at index."""
		pass

	@abc.abstractmethod
	def __len__(self):
		r"""The number of examples in the dataset."""
		pass

	def __repr__(self):
		return (
			f'Dataset("{self.name}", num_graphs={len(self)},'
			+ f" save_path={self.save_path})"
		)


class BasePCGraph(MyDGLDataset):
	'''
	Given a pc dataset, save it as a graph dataset.
	on_the_fly: '0' pre_compute graph structure with basis information
				'basis' pre_compute graph structure with basis information
	in_memory: save all data in memory. Only works for small dataset.
	'''
	def __init__(self, pc_dataset=None,
	             dataset_root=None,
	             dataset_name='',
	             param_str=None,
	             # radius=-1,
	             max_deg=2,
	             on_the_fly='simple',
	             force_reload=False,
	             minimal_edge_len=-1,
	             k_neighbor=5, use_knn=True,
	             in_memory=False
	             ):

		self.on_the_fly = on_the_fly
		if on_the_fly == '0':
			self.pre_compute = True
		elif on_the_fly == 'simple' or on_the_fly == 'basis':
			self.pre_compute = False
		else:
			raise NotImplementedError
		# self.radius = radius
		self.max_deg = max_deg
		self.minimal_edge_len = minimal_edge_len

		self.k_neighbor = k_neighbor
		self.use_knn = use_knn

		self.pc_dataset = pc_dataset
		self.param_str = param_str
		self.processed = os.path.join(dataset_root, 'processed', param_str)
		self.allname_file = os.path.join(self.processed, 'All_name.txt')
		self.trans_file = os.path.join(self.processed, 'trans.npy')
		self.in_memory = in_memory
		if in_memory:
			self.all_data_list = []
		super().__init__(name=dataset_name, force_reload=force_reload, pre_compute=self.pre_compute)


	def has_cache(self):
		return os.path.isdir(self.processed)

	def process(self):
		if not os.path.isdir(self.processed):
			os.makedirs(self.processed)
		all_name_file = open(self.allname_file, 'w')
		self.all_name_list = []
		print(f'Processing graph in setting {self.param_str}')
		all_transform = []
		for i in tqdm(range(len(self.pc_dataset))):
			pc = self.pc_dataset[i]
			graph_ref, graph_src = convert_to_graph(pc, max_deg=self.max_deg,
			                                        k_neighbor=self.k_neighbor, use_knn=self.use_knn, type='basis')
			# print(f'Processing graph {i}')
			save_name_i = os.path.join(self.processed, f"{i}")
			dgl.save_graphs(save_name_i + '_ref_src.dgl', [graph_ref, graph_src])
			self.all_data_list.append([graph_ref, graph_src])
			all_transform.append(pc['transform'])
			self.all_name_list.append(save_name_i)

		self.all_tranform = np.stack(all_transform, axis=0)
		np.save(self.trans_file, self.all_tranform)
		all_name_file.write('\n'.join(self.all_name_list))
		all_name_file.close()

	def load(self):
		self.all_name_ = open(self.allname_file)
		self.all_name_list = [i.strip() for i in self.all_name_.readlines()]
		self.all_tranform = np.load(self.trans_file)
		if self.in_memory:
			print(f'Loading graphs into memory..')
			for n in tqdm(self.all_name_list):
				self.all_data_list.append(dgl.load_graphs(n + '_ref_src.dgl')[0])

	def __getitem__(self, i):
		if not self.pre_compute:
			pc = self.pc_dataset[i]
			graph_ref, graph_src = convert_to_graph(pc, max_deg=self.max_deg,
			                                        k_neighbor=self.k_neighbor, use_knn=self.use_knn, type=self.on_the_fly)
			return graph_ref, graph_src, pc['transform']
		else:
			if self.in_memory:
				g_ref, g_src = self.all_data_list[i]
			else:
				g_ref, g_src = dgl.load_graphs(self.all_name_list[i] + '_ref_src.dgl')[0]
			return g_ref, g_src, self.all_tranform[i]


	def __len__(self):
		return len(self.pc_dataset)
