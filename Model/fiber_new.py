import einops
import torch


class Fiber(object):
	"""A dict for features structures"""
	def __init__(self, fiber=None):
		"""
		define fiber structure.
		:param fiber_size: e.g. {(0,1): 2, (0,2): 2, (1, 1): (2)}
		keys are degrees of the 1st and the 2nd component, and values are multiplicities
		tensor of this type is of shape (point, mutiplicity, degree)
		"""

		# sorted by degrees
		self.structure = dict(sorted(fiber.items()))

		self.indices = {}
		self.length = []
		idx = 0
		for d in self.structure.keys():
			length = (int(d[0]) * 2 + 1) * (int(d[1]) * 2 + 1)
			self.indices[d] = (idx, idx + length)
			self.length.append(length)
			idx += length
		self.deg_size = idx

		# when all mulicities are the same, we fuse all fibers to a big tensor.
		multi_all = set(self.structure.values())
		if len(multi_all) == 1:
			self.can_fuse = True
			self.shared_channel = list(multi_all)[0]
		else:
			self.can_fuse = False

	def __repr__(self):
		return f"{self.structure}"

	def items(self):
		return self.structure.items()

	def keys(self):
		return self.structure.keys()

	def fuse(self, tensor):
		if type(tensor) == torch.Tensor:
			return tensor
		if self.can_fuse:
			if not set(self.keys()) == set(tensor.keys()):
				raise ValueError(f'Can not fuse features: {self.keys()} != {tensor.keys()}')
			All_list = []
			for k in self.keys():
				All_list.append(tensor[k])
			All_tensor = torch.cat(All_list, -1)
			return All_tensor
		else:
			raise ValueError(f'Non-fusable fibers.')

	def cat(self, y, complete=False):
		# concatenate fiber y to the current fiber
		# add channel numbers for shared degrees
		# complete: whether to create new degrees that are in y and not in self.
		New_dict = {}
		for k in self.keys():
			New_dict[k] = self.structure[k]
			if k in y.keys():
				New_dict[k] += y.structure[k]

		# add new keys
		if complete == True:
			for k in y.keys():
				if k not in self.keys():
					New_dict = y.structure[k]
		New_fiber = Fiber(fiber=New_dict)
		return New_fiber


class Tensor_fiber:
	# define a tensor of a specific Fiber type
	def __init__(self, fiber=None, tensor=None):
		# two ways to initialize:
		# 1. tensor is a dict with keys=degree, values=tensor, fiber is None
		# 2. fiber is Fiber, tensor is Tensor

		if fiber is None:
			assert type(tensor) == dict, 'Need a tensor as a dict.'
			# consistent shape
			for d, t in tensor.items():
				assert t.shape[-1] == (2 * d[0] + 1) * (2 * d[1] + 1), f'Tensor shape invalid: shape of deg{d} is {t.shape[-1]}'
			New_Dict = {}
			for d, tensor_d in tensor.items():
				New_Dict[d] = tensor_d.shape[-2] #mutiplicity
			self.fiber = Fiber(New_Dict)
			if self.fiber.can_fuse:
				self.tensor = self.fiber.fuse(tensor)
			else:
				self.tensor = tensor
		elif type(fiber) == Fiber:
			assert type(tensor) == torch.Tensor, 'Need a tensor as a pytorch Tensor'
			assert tensor.ndim == 3, 'Need a 3 dimensional tensor (batch, channel, degree)'
			assert fiber.deg_size == tensor.shape[-1], f'Shape does not match Fiber dim {fiber.deg_size} != Tensor dim {tensor.shape[-1]}'
			self.fiber = fiber
			self.tensor = tensor
		else:
			raise ValueError('fiber must be None or class Fiber.')

		self.can_fuse = self.fiber.can_fuse

	def get_all_component(self):
		if self.can_fuse:
			Tensor_dict = {}
			tensor_list = torch.split(self.tensor, self.fiber.length, -1)
			for d, t in zip(self.fiber.keys(), tensor_list):
				Tensor_dict[d] = t
			return Tensor_dict
		else:
			return self.tensor

	def get_component(self, deg):
		if not self.can_fuse:
			return self.tensor[deg]
		else:
			return self.tensor[:,:, self.fiber.indices[deg][0]:self.fiber.indices[deg][1]]

	def cat_tensor(self, tensor_fiber_y):
		# concate tensor_y belonging to fiber_y to tensor_x belonging to current fiber
		# along the mutiplicity dim
		# if both tensor are fused, simply cat them in dim channel
		if self.can_fuse and tensor_fiber_y.can_fuse and self.fiber.keys() == tensor_fiber_y.fiber.keys():
			new_tensor = torch.cat([self.tensor, tensor_fiber_y.tensor], 1)
			new_fiber = self.fiber.cat(tensor_fiber_y.fiber)
			return Tensor_fiber(fiber=new_fiber, tensor=new_tensor)
		# otherwise cat each element
		component_x = self.get_all_component()
		component_y = tensor_fiber_y.get_all_component()
		new_component = {}
		for d in self.fiber.keys():
			if d in tensor_fiber_y.fiber.keys():
				new_component[d] = torch.cat([component_x[d], component_y[d]], 1)
			else:
				new_component[d] = component_x[d]
		return Tensor_fiber(tensor=new_component)

	def add_tensor(self, tensor_fiber_y):
		# add tensor_y if possible, otherwise simply keep tensor_x
		# if both tensor are fused, simply add them
		if self.can_fuse and tensor_fiber_y.can_fuse and self.fiber.keys() == tensor_fiber_y.fiber.keys() \
				and self.tensor.shape[1] == tensor_fiber_y.tensor.shape[1]:
			new_tesnor = tensor_fiber_y.tensor + self.tensor
			return Tensor_fiber(fiber=self.fiber, tensor=new_tesnor)
		# otherwise add each plausible element
		component_x = self.get_all_component()
		component_y = tensor_fiber_y.get_all_component()
		new_component = {}
		for d in self.fiber.keys():
			if d in tensor_fiber_y.fiber.keys() and self.fiber.structure[d] == tensor_fiber_y.fiber.structure[d]:
				new_component[d] = component_x[d] + component_y[d]
			else:
				new_component[d] = component_x[d]
		return Tensor_fiber(tensor=new_component)



	def sep_head(self, head):
		# divide mutiplicity into 'head' groups for computing attentions
		# all tensor are concatenated in dim -1
		# return tensor
		if self.can_fuse:
			return einops.rearrange(self.tensor, 'p (h m) d -> p h (m d)', h=head)
		else:
			return torch.cat([einops.rearrange(i, 'p (h m) d -> p h (m d)', h=head) for i in self.tensor.values()], -1)

	def get_sub_tensor(self, fiber_y):
		# if fiber_y and self are both fused tensor and their keys are the same.
		if self.can_fuse and fiber_y.can_fuse and self.fiber.keys() == fiber_y.keys():
			assert self.fiber.shared_channel > fiber_y.shared_channel, f'Required channel is too large.'
			fiber_x = Fiber({d: m-fiber_y.shared_channel for d, m in self.fiber.items()})
			tensor_y = self.tensor[:, -fiber_y.shared_channel:, :]
			tensor_x = self.tensor[:, :-fiber_y.shared_channel, :]
			return Tensor_fiber(fiber=fiber_x, tensor=tensor_x), Tensor_fiber(fiber=fiber_y, tensor=tensor_y)
		# Otherwise iterates over the dict
		y_dict = {}
		x_dict = {}
		for d in self.fiber.keys():
			if d in fiber_y.keys():
				assert self.fiber.structure[d] > fiber_y.structure[d], f'Required channel is too large.'
				y_dict[d] = self.get_component(d)[:, -fiber_y.structure[d]:, :]
				if self.fiber.structure[d] > fiber_y.structure[d]:
					x_dict[d] = self.get_component(d)[:, :-fiber_y.structure[d], :]
			else:
				x_dict[d] = self.get_component(d)
		return Tensor_fiber(tensor=x_dict), Tensor_fiber(tensor=y_dict)


	def inserttoG(self, G):
		# put a tensor fiber into the underlying graph structure, so that
		# it can be read out easily
		Component = self.get_all_component()
		for k, v in Component.items():
			assert f'Tensor_{k}' not in G.edata.keys(), f'Key Tensor_{k} already exists in graph!'
			G.ndata[f'Tensor_{k}'] = v


def fiber_merge(fiber1, fiber2, type=None):
	'''
	(0, 1), (1, 0) -> (1, 0), (0, 1), (1, 1)
	'''
	if type == '11':
		New_fiber_dict = {}
		for k1 in fiber1.keys():
			m1 = fiber1.structure[k1]
			for k2 in fiber2.keys():
				m2 = fiber2.structure[k2]
				if k1[0] == 0 or k2[0] == 0:
					assert m1 == m2, 'Cannot merge tensors due to channel difference.'
					new_deg = (k1[0], k2[0])
					New_fiber_dict[new_deg] = m1
				elif k1[0] == 1 or k2[0] == 1:
					assert m1 == m2, 'Cannot merge tensors due to channel difference.'
					New_fiber_dict[(1, 1)] = m1
		return Fiber(fiber=New_fiber_dict)
	else:
		raise NotImplementedError


def generate_fiber(max_degree_total, max_degree_side, num_channels):
	Inner_fiber = {}
	for i in range(min(max_degree_total + 1, max_degree_side[0] + 1)):
		for j in range(min(max_degree_total - i + 1, max_degree_side[1] + 1)):
			Inner_fiber.update({(i, j): num_channels})
	return Fiber(fiber=Inner_fiber)