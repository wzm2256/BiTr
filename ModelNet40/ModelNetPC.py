import os.path
import os.path as osp
import pdb
from typing import Dict, Optional

import numpy as np
import torch.utils.data

try:
    import open3d as o3d
except:
    print('Can not import open3d. No key points selection and no visualization.')

# from IPython import embed
import pickle
from .utils.common import load_pickle
from .utils.pointcloud import random_sample_transform, apply_transform, inverse_transform, regularize_normals
from .utils.registration import compute_overlap
# from .utils.open3d import estimate_normals, voxel_downsample
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


class ModelNetPairDataset(torch.utils.data.Dataset):
    # fmt: off
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
        'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
        'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'curtain', 'desk', 'door', 'dresser',
        'glass_box', 'guitar', 'keyboard', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_INDICES = [
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36,
        38, 39
    ]

    ALL_CATEGORIES_ShapeNet = ['airplane', 'bathtub', 'bed', 'bottle', 'cap', 'car', 'chair', 'guitar', 'helmet',
                               'knife', 'laptop', 'motorcycle', 'mug', 'skateboard', 'table', 'vessel']

    # fmt: on

    def __init__(
        self,
        dataset_root: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = None,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: Optional[float] = None,
        keep_ratio: Optional[float] = None,
        crop_method: str = 'plane',
        asymmetric: bool = True,
        class_indices: str = 'all',
        deterministic: bool = False,
        twice_sample: bool = True,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
        use_random=True,
        type='ShapeNet'
    ):
        super(ModelNetPairDataset, self).__init__()

        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.subset = subset
        self.type = type
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.asymmetric = asymmetric
        self.class_indices = self.get_class_indices(class_indices, asymmetric)
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index
        self.use_random_sample = use_random

        data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))


        if len(self.class_indices) != 2:
            data_list = [x for x in data_list if x['label'] in self.class_indices]
            if overfitting_index is not None:
                data_list = [data_list[overfitting_index]]
            self.data_list = data_list
            self.intra = False
        else:
            self.intra = True

            if subset == 'train' and overfitting_index is not None:
                # pdb.set_trace()
                self.data_list1 = [x for x in data_list if x['label'] == self.class_indices[0]][:int(overfitting_index)]
                self.data_list2 = [x for x in data_list if x['label'] == self.class_indices[1]][:int(overfitting_index)]
            else:
                self.data_list1 = [x for x in data_list if x['label'] == self.class_indices[0]]
                self.data_list2 = [x for x in data_list if x['label'] == self.class_indices[1]]

            #
            # data_list3 = [x for x in data_list if x['label'] == self.class_indices[0]]
            # data_list4 = [x for x in data_list if x['label'] == self.class_indices[1]]


            # pdb.set_trace()

        # pdb.set_trace()
    def get_class_indices(self, class_indices, asymmetric):
        r"""Generate class indices.
        'all' -> all 40 classes.
        'seen' -> first 20 classes.
        'unseen' -> last 20 classes.
        list|tuple -> unchanged.
        asymmetric -> remove symmetric classes.
        """
        if self.type == 'ShapeNet':
            if ',' in class_indices:
                class_indices = [self.ALL_CATEGORIES_ShapeNet.index(i) for i in class_indices.split(',')]
                return class_indices

            single_index = self.ALL_CATEGORIES_ShapeNet.index(class_indices)
            class_indices = [single_index]
            return class_indices

        if isinstance(class_indices, str):
            # assert class_indices in ['all', 'seen', 'unseen']
            if class_indices == 'all':
                class_indices = list(range(40))
            elif class_indices == 'seen':
                class_indices = list(range(20))
            elif class_indices == 'unseen':
                class_indices = list(range(20, 40))
            else:
                single_index = self.ALL_CATEGORIES.index(class_indices)
                class_indices = [single_index]
                return class_indices
        if asymmetric:
            class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
        return class_indices

    def __getitem__(self, index):
        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # pdb.set_trace()
        if not self.intra:
            if self.overfitting_index is not None:
                index = 0

            data_dict: Dict = self.data_list[index]
            raw_points = data_dict['points'].copy()
            raw_normals = data_dict['normals'].copy()
            label = data_dict['label']

            # normalize raw point cloud
            raw_points = normalize_points(raw_points)

            #######################
            # from .utils.open3d import estimate_normals, voxel_downsample
            # raw_points, raw_normals = voxel_downsample(raw_points, 0.05, normals=raw_normals)
            #######################

            # once sample on raw point cloud
            # pdb.set_trace()
            if not self.twice_sample:
                raw_points, raw_normals = random_sample_points(raw_points, self.num_points, normals=raw_normals, use_random=self.use_random_sample)

            # split reference and source point cloud
            ref_points = raw_points.copy()
            ref_normals = raw_normals.copy()

            # twice transform
            if self.twice_transform:
                transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
                ref_points, ref_normals = apply_transform(ref_points, transform, normals=ref_normals)

            src_points = ref_points.copy()
            src_normals = ref_normals.copy()

            # random transform to source point cloud
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            inv_transform = inverse_transform(transform)
            src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

            raw_ref_points = ref_points
            raw_ref_normals = ref_normals
            raw_src_points = src_points
            raw_src_normals = src_normals

            while True:
                ref_points = raw_ref_points
                ref_normals = raw_ref_normals
                src_points = raw_src_points
                src_normals = raw_src_normals
                # crop
                if self.keep_ratio is not None:
                    if self.crop_method == 'plane':
                        ref_points, ref_normals = random_crop_point_cloud_with_plane(
                            ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                        )
                        src_points, src_normals = random_crop_point_cloud_with_plane(
                            src_points, keep_ratio=self.keep_ratio, normals=src_normals
                        )
                    else:
                        viewpoint = random_sample_viewpoint()
                        ref_points, ref_normals = random_crop_point_cloud_with_point(
                            ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                        )
                        src_points, src_normals = random_crop_point_cloud_with_point(
                            src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                        )

                # data check
                is_available = True
                # check overlap
                if self.check_overlap:
                    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                    if self.min_overlap is not None:
                        is_available = is_available and overlap >= self.min_overlap
                    if self.max_overlap is not None:
                        is_available = is_available and overlap <= self.max_overlap
                if is_available:
                    break

            if self.twice_sample:
                # pdb.set_trace()
                # twice sample on both point clouds
                if self.keep_ratio is None:
                    resample_point = self.num_points
                else:
                    resample_point = int(self.num_points * self.keep_ratio)
                ref_points, ref_normals = random_sample_points(ref_points, resample_point, normals=ref_normals,
                                                               use_random=self.use_random_sample)
                src_points, src_normals = random_sample_points(src_points, resample_point, normals=src_normals,
                                                               use_random=self.use_random_sample)

            # random jitter
            if self.noise_magnitude is not None:
                ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
                src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

            # random shuffle
            ################
            # ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
            # src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)
            ################

            if self.voxel_size is not None:
                from .utils.open3d import estimate_normals, voxel_downsample
                # voxel downsample reference point cloud
                ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
                src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

            new_data_dict = {
                'raw_points': raw_points.astype(np.float32),
                'ref_points': ref_points.astype(np.float32),
                'src_points': src_points.astype(np.float32),
                'transform': transform.astype(np.float32),
                'label': int(label),
                'index': int(index),
            }

            if self.estimate_normal:
                from .utils.open3d import estimate_normals, voxel_downsample
                ref_normals = estimate_normals(ref_points)
                ref_normals = regularize_normals(ref_points, ref_normals)
                src_normals = estimate_normals(src_points)
                src_normals = regularize_normals(src_points, src_normals)

            if self.return_normals:
                new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
                new_data_dict['ref_normals'] = ref_normals.astype(np.float32)
                new_data_dict['src_normals'] = src_normals.astype(np.float32)

            if self.return_occupancy:
                new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
                new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

            # pdb.set_trace()
            return new_data_dict

        else:

            if self.subset == 'train':
                j = np.random.randint(0, len(self.data_list1))
                i = np.random.randint(0, len(self.data_list2))
            else:
                i = index
                j = index

            # i = index // len(self.data_list1)
            # j = index - i * len(self.data_list1)

            # pdb.set_trace()
            src_dict = self.data_list1[j]
            ref_dict = self.data_list2[i]

            src_raw_points = src_dict['points'].copy()
            ref_raw_points = ref_dict['points'].copy()

            src_raw_normal = src_dict['normals'].copy()
            ref_raw_normal = ref_dict['normals'].copy()

            scale = 0.6
            src_raw_points = normalize_points(src_raw_points)
            ref_raw_points = normalize_points(ref_raw_points) * scale


            src_raw_points, src_raw_normal = random_sample_points(src_raw_points, self.num_points, normals=src_raw_normal,
                                                               use_random=self.use_random_sample)

            ref_raw_points, ref_raw_normal = random_sample_points(ref_raw_points, self.num_points, normals=ref_raw_normal,
                                                               use_random=self.use_random_sample)

            transform_ref = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            ref_points, ref_normals = apply_transform(ref_raw_points, transform_ref, normals=ref_raw_normal)

            # ref_points += np.expand_dims(move_ref, 0)
            # pdb.set_trace()

            transform_src = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            inv_transform_src = inverse_transform(transform_src)
            src_points, src_normals = apply_transform(src_raw_points, inv_transform_src, normals=src_raw_normal)

            ref_points, ref_normals = random_crop_point_cloud_with_plane(ref_points, keep_ratio=self.keep_ratio, normals=ref_normals)
            src_points, src_normals = random_crop_point_cloud_with_plane(src_points, keep_ratio=self.keep_ratio, normals=src_normals)

            if self.noise_magnitude is not None:
                ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
                src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

            # pdb.set_trace()
            move_ref = np.array([1., 0., 0.]).astype(np.float32)
            move_aside = np.eye(4, dtype=np.float32)
            move_aside[:3, 3] = move_ref.astype(np.float32)
            # pdb.set_trace()
            new_data_dict = {
                'raw_points': None,
                'ref_points': ref_points.astype(np.float32),
                'src_points': src_points.astype(np.float32),
                'transform':  (move_aside @ transform_ref @ transform_src).astype(np.float32),
                'label': 0,
                'index': int(index),
            }

            if self.return_normals:
                new_data_dict['raw_normals'] = None
                new_data_dict['ref_normals'] = ref_normals.astype(np.float32)
                new_data_dict['src_normals'] = src_normals.astype(np.float32)
            return new_data_dict



    def __len__(self):
        # if self.intra == True:
        #     return len(self.data_list1) * len(self.data_list2)
        if self.intra and self.subset == 'train':
            return 1000
        elif self.intra and self.subset == 'test':
            return min(len(self.data_list1), len(self.data_list2))

        return len(self.data_list)



if __name__ == '__main__':
    dataset = ModelNetPairDataset(
    # dataset_root= 'Data/ModelNet40',
    dataset_root='Data/ShapeNet',
    subset= 'test',
    # num_points: int = 1024,
    # voxel_size: Optional[float] = None,
    rotation_magnitude = 0.0,
    translation_magnitude=0.0,
    # noise_magnitude: Optional[float] = None,
    keep_ratio=0.7,
    # crop_method: str = 'plane',
    # asymmetric: bool = True,
    class_indices = 'airplane',
    # class_indices='car,motorcycle',
    # deterministic: bool = False,
    # twice_sample: bool = False,
    # twice_transform: bool = False,
    # return_normals: bool = True,
    # return_occupancy: bool = False,
    # min_overlap: Optional[float] = None,
    # max_overlap: Optional[float] = None,
    # estimate_normal: bool = False,
    # overfitting_index: Optional[int] = None,
    # use_random = True,
    type='ShapeNet'
    )
    import utils

    # print(len(d))
    # pdb.set_trace()
#     bb = BBPair('Data/BB/', subset='train', category='Bowl', return_normals=True)
#
#     a = bb[1]
    pdb.set_trace()
    for i, data in enumerate(dataset):
        # print(i)
        # pdb.set_trace()
        source_PC = data['src_points']
        Ref_PC = data['ref_points']
        transform = data['transform']
        gt_source = source_PC @ transform[:3, :3].T + transform[:3, 3:].T

        # pdb.set_trace()
        utils.vis_PC([source_PC, Ref_PC], point_size=5.)
        utils.vis_PC([gt_source, Ref_PC], point_size=5.)

# class MugPair(torch.utils.data.Dataset):
#
#     def __init__(
#         self,
#         dataset_root: str,
#         subset: str = 'train',
#         get_color=False,
#         num_points=64,
#         task='mug'
#     ):
#         super(MugPair, self).__init__()
#
#         assert task == 'mug'
#         self.num_points = num_points
#         if subset == 'test':
#             ####################
#             self.class_indices = [6, 9, 10]
#             self.max_sample = 10
#             ###################
#             # self.class_indices = [0, 1, 4, 6, 9, 10]
#             # self.max_sample = 10
#             ###################
#         elif subset == 'train':
#             self.class_indices = [0,1,4]
#             self.max_sample = 10
#         else:
#             raise NotImplementedError
#             # self.class_indices = [int(class_indices)]
#
#         self.dataset_root = dataset_root
#         # self.subset = dataset_root
#         # self.class_indices = class_indices
#         self.get_color = get_color
#
#         self.dataset_root_new = 'Data\manipulate'
#
#         self.new_data_list = []
#         for i in self.class_indices:
#             with open(os.path.join(self.dataset_root_new, task, f'{task}_{i}.pk'), 'rb') as f:
#                 # pdb.set_trace()
#                 data_class = pickle.load(f)
#                 self.new_data_list += data_class
#
#         self.data_list = []
#         for i in self.class_indices:
#             # pdb.set_trace()
#             folder = os.path.join(dataset_root, f'mug{i}')
#             all_file = os.listdir(folder)
#             max_ind = 0
#             for n in all_file:
#                 if 'Ref_' in n:
#                     current_ind = int(n.split('_')[-1].split('.')[0])
#                     if current_ind > max_ind:
#                         max_ind = current_ind
#             scale = 30.
#             for pair_id in range(min(max_ind, self.max_sample)):
#
#                 R = np.load(os.path.join(folder, f'R_{pair_id + 1}.npy')).astype(np.float32)
#                 t = np.load(os.path.join(folder, f't_{pair_id + 1}.npy')).astype(np.float32) / scale
#                 Src = np.load(os.path.join(folder, f'Ref_{pair_id + 1}.npy')).astype(np.float32) / scale
#                 Src_color = np.load(os.path.join(folder, f'Ref_color_{pair_id + 1}.npy')).astype(np.float32)
#                 Ref = np.load(os.path.join(folder, f'Src_{pair_id + 1}.npy')).astype(np.float32) / scale
#                 Ref_color = np.load(os.path.join(folder, f'Src_color_{pair_id + 1}.npy')).astype(np.float32)
#
#                 transform = np.eye(4, dtype=np.float32)
#                 transform[:3, :3] = R
#                 transform[:3, 3] = t
#
#                 self.data_list.append({'transform': transform,
#                                         'src_points':Src,
#                                         'Src_color': Src_color,
#                                         'ref_points': Ref,
#                                         'Ref_color': Ref_color})
#
#     def __getitem__(self, index):
#
#         data_dict = self.data_list[index]
#
#         if self.get_color:
#             ref_points, ref_colors = random_sample_points(data_dict['ref_points'], self.num_points, color=data_dict['Ref_color'], normals=None, use_random=True)
#             src_points, src_colors = random_sample_points(data_dict['src_points'], self.num_points, color=data_dict['Src_color'], normals=None, use_random=True)
#         else:
#             ref_points = random_sample_points(data_dict['ref_points'], self.num_points, normals=None, use_random=True)
#             src_points = random_sample_points(data_dict['src_points'], self.num_points, normals=None, use_random=True)
#
#         data_item={}
#         data_item['transform'] = data_dict['transform']
#         data_item['ref_points'] = ref_points
#         data_item['src_points'] = src_points
#         data_item['index'] = index
#
#         if self.get_color:
#             data_item['transform'] = data_dict['transform']
#             data_item['Ref_color'] = ref_colors * 0.5 + 0.5
#             data_item['Src_color'] = src_colors * 0.5 + 0.5
#
#         return data_item
#
#     def __len__(self):
#         return len(self.data_list)



#
# class ModelNetAlign(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         dataset_root: str,
#         subset: str,
#         num_points: int = 1024,
#         voxel_size: Optional[float] = None,
#         rotation_magnitude: float = 45.0,
#         translation_magnitude: float = 0.5,
#         noise_magnitude: Optional[float] = None,
#         keep_ratio: Optional[float] = None,
#         crop_method: str = 'plane',
#         asymmetric: bool = True,
#         class_indices: str = 'all',
#         deterministic: bool = False,
#         twice_sample: bool = False,
#         twice_transform: bool = False,
#         return_normals: bool = True,
#         return_occupancy: bool = False,
#         min_overlap: Optional[float] = None,
#         max_overlap: Optional[float] = None,
#         estimate_normal: bool = False,
#         overfitting_index: Optional[int] = None,
#         use_random=True,
#         maximal_sample=5,
#         use_key_points=False
#     ):
#         super(ModelNetAlign, self).__init__()
#
#         assert subset in ['train', 'val', 'test']
#         assert crop_method in ['plane', 'point']
#         self.maximal_sample = maximal_sample
#         self.dataset_root = dataset_root
#         self.subset = subset
#
#         self.num_points = num_points
#         self.voxel_size = voxel_size
#         self.rotation_magnitude = rotation_magnitude
#         self.translation_magnitude = translation_magnitude
#         self.noise_magnitude = noise_magnitude
#         self.keep_ratio = keep_ratio
#         self.crop_method = crop_method
#         self.asymmetric = asymmetric
#         self.deterministic = deterministic
#         self.twice_sample = twice_sample
#         self.twice_transform = twice_transform
#         self.return_normals = return_normals
#         self.return_occupancy = return_occupancy
#         self.min_overlap = min_overlap
#         self.max_overlap = max_overlap
#         self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
#         self.estimate_normal = estimate_normal
#         self.overfitting_index = overfitting_index
#         self.use_random_sample = use_random
#         self.use_key_points = use_key_points
#         # data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))
#         class_indice_list = class_indices.split('_')
#
#         Data_dict = {}
#         for c in class_indice_list:
#             Data_dict[c] = []
#             path_all = os.path.join(dataset_root, c, subset)
#             for pc in sorted(os.listdir(path_all)):
#                 data = np.load(os.path.join(path_all, pc))
#                 Data_dict[c].append(data)
#         self.Data_dict = Data_dict
#
#     def __getitem__(self, index):
#         if self.overfitting_index is not None:
#             index = 0
#
#         i = index // self.maximal_sample
#         j = index - i * self.maximal_sample
#
#         All_keys = list(self.Data_dict.keys())
#
#         # pdb.set_trace()
#         if len(All_keys) == 2:
#             raw_points_1 = self.Data_dict[All_keys[0]][i]
#             raw_points_2 = self.Data_dict[All_keys[1]][j]
#             print(f'{All_keys[0]} - {i} ------- {All_keys[1]} - {j}')
#         elif len(All_keys) == 1:
#             raw_points_1 = self.Data_dict[All_keys[0]][i]
#             raw_points_2 = self.Data_dict[All_keys[0]][j]
#             print(f'{All_keys[0]} - {i} ------- {All_keys[0]} - {j}')
#         else:
#             raise ValueError('class length not correct')
#
#         if len(All_keys) == 2 and All_keys[0] == 'chair' and All_keys[1] == 'bed':
#             scale = (0.4, 1.0)
#             raw_points_1, raw_points_2 = normalize_points_two(raw_points_1, raw_points_2, scale)
#             transform = np.eye(4)
#             x_max = np.max(raw_points_2, axis=0)[0]
#             y_max = np.max(raw_points_2, axis=0)[1]
#             delta_x = x_max - np.min(raw_points_1, axis=0)[0]
#             delta_y = y_max - np.max(raw_points_1, axis=0)[1]
#             transform[0:3, 3]=np.array([delta_x, delta_y, 0])
#         # elif All_keys[0] == 'car' and All_keys[1] == 'airplane':
#         #     scale = (0.4, 1.0)
#         #     raw_points_1, raw_points_2 = normalize_points_two(raw_points_1, raw_points_2, scale)
#         #     transform = np.eye(4)
#         #     x_max = np.max(raw_points_2, axis=0)[0]
#         #     y_max = np.max(raw_points_2, axis=0)[1]
#         #     delta_x = x_max - np.min(raw_points_1, axis=0)[0]
#         #     delta_y = y_max - np.max(raw_points_1, axis=0)[1]
#         #     transform[0:3, 3]=np.array([delta_x, delta_y, 0])
#         else:
#             scale = (1., 1.)
#             raw_points_1, raw_points_2 = normalize_points_two(raw_points_1, raw_points_2, scale)
#             transform = np.eye(4)
#
#
#
#
#         src_points = random_sample_points(raw_points_1, self.num_points, normals=None, use_random=self.use_random_sample)
#         ref_points = random_sample_points(raw_points_2, self.num_points, normals=None, use_random=self.use_random_sample)
#
#         if self.noise_magnitude is not None:
#             ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
#             src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)
#
#         if self.use_key_points:
#             ref_points_pcd = o3d.geometry.PointCloud()
#             ref_points_pcd.points = o3d.utility.Vector3dVector(ref_points)
#
#             src_points_pcd = o3d.geometry.PointCloud()
#             src_points_pcd.points = o3d.utility.Vector3dVector(src_points)
#
#             ref_points_pcd_down = o3d.geometry.keypoint.compute_iss_keypoints(ref_points_pcd, salient_radius=0.04, non_max_radius=0.04, gamma_21=0.8, gamma_32=0.8)
#             src_points_pcd_down = o3d.geometry.keypoint.compute_iss_keypoints(src_points_pcd, salient_radius=0.04, non_max_radius=0.04, gamma_21=0.8, gamma_32=0.8)
#
#             # ref_points_pcd_down = o3d.geometry.keypoint.compute_iss_keypoints(ref_points_pcd)
#             # src_points_pcd_down = o3d.geometry.keypoint.compute_iss_keypoints(src_points_pcd)
#
#             ref_points = np.asarray(ref_points_pcd_down.points)
#             src_points = np.asarray(src_points_pcd_down.points)
#
#             print(f'Num points {ref_points.shape[0]}-{src_points.shape[0]}')
#             # pdb.set_trace()
#
#         new_data_dict = {
#             # 'raw_points': raw_points.astype(np.float32),
#             'ref_points': ref_points.astype(np.float32),
#             'src_points': src_points.astype(np.float32),
#             'transform': transform.astype(np.float32),
#             # 'label': int(label),
#             'index': int(index),
#         }
#
#
#         return new_data_dict
#
#     def __len__(self):
#         return self.maximal_sample ** 2
#
# class ShapeNet(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         dataset_root: str,
#         subset: str,
#         num_points: int = 1024,
#         voxel_size: Optional[float] = None,
#         rotation_magnitude: float = 45.0,
#         translation_magnitude: float = 0.5,
#         noise_magnitude: Optional[float] = None,
#         keep_ratio: Optional[float] = None,
#         crop_method: str = 'plane',
#         asymmetric: bool = True,
#         class_indices: str = 'all',
#         deterministic: bool = False,
#         twice_sample: bool = False,
#         twice_transform: bool = False,
#         return_normals: bool = True,
#         return_occupancy: bool = False,
#         min_overlap: Optional[float] = None,
#         max_overlap: Optional[float] = None,
#         estimate_normal: bool = False,
#         overfitting_index: Optional[int] = None,
#         use_random=True,
#         maximal_sample=5,
#         use_key_points=False
#     ):
#         super(ShapeNet, self).__init__()
#
#         assert subset in ['train', 'test']
#         assert crop_method in ['plane', 'point']
#         self.maximal_sample = maximal_sample
#         self.dataset_root = dataset_root
#         self.subset = subset
#
#         self.num_points = num_points
#         self.voxel_size = voxel_size
#         self.rotation_magnitude = rotation_magnitude
#         self.translation_magnitude = translation_magnitude
#         self.noise_magnitude = noise_magnitude
#         self.keep_ratio = keep_ratio
#         self.crop_method = crop_method
#         self.asymmetric = asymmetric
#         self.deterministic = deterministic
#         self.twice_sample = twice_sample
#         self.twice_transform = twice_transform
#         self.return_normals = return_normals
#         self.return_occupancy = return_occupancy
#         self.min_overlap = min_overlap
#         self.max_overlap = max_overlap
#         self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
#         self.estimate_normal = estimate_normal
#         self.overfitting_index = overfitting_index
#         self.use_random_sample = use_random
#         self.use_key_points = use_key_points
#         # data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))
#         class_indice_list = class_indices.split('_')
#         self.class_indice_list = class_indice_list
#         if subset == 'train':
#             start_index = 0
#         else:
#             start_index = 100
#
#         Data_dict = {}
#         for c in class_indice_list:
#             Data_dict[c] = []
#             Data_dict[f'{c}_key'] = []
#             path_all = os.path.join(dataset_root, c)
#             for i in range(min(maximal_sample, 100)):
#                 data = np.loadtxt(os.path.join(path_all, f'{start_index + i}.txt'))
#                 key = np.loadtxt(os.path.join(path_all, f'{start_index + i}_key.txt'))
#                 Data_dict[c].append(data)
#                 Data_dict[f'{c}_key'].append(key)
#         self.Data_dict = Data_dict
#
#     def __getitem__(self, index):
#         if self.overfitting_index is not None:
#             index = 0
#
#         i = index // self.maximal_sample
#         j = index - i * self.maximal_sample
#
#
#         if self.use_key_points:
#             if len(self.class_indice_list) == 2:
#                 src_points = self.Data_dict[f'{self.class_indice_list[0]}_key'][i]
#                 ref_points = self.Data_dict[f'{self.class_indice_list[1]}_key'][j]
#                 print(f'{self.class_indice_list[0]} - {i} ------- {self.class_indice_list[1]} - {j}')
#             elif len(self.class_indice_list) == 1:
#                 src_points = self.Data_dict[f'{self.class_indice_list[0]}_key'][i]
#                 ref_points = self.Data_dict[f'{self.class_indice_list[0]}_key'][j]
#                 print(f'{self.class_indice_list[0]} - {i} ------- {self.class_indice_list[0]} - {j}')
#             else:
#                 raise ValueError('class length not correct')
#         else:
#             if len(self.class_indice_list) == 2:
#                 src_points = self.Data_dict[f'{self.class_indice_list[0]}'][i]
#                 ref_points = self.Data_dict[f'{self.class_indice_list[1]}'][j]
#                 print(f'{self.class_indice_list[0]} - {i} ------- {self.class_indice_list[1]} - {j}')
#             elif len(self.class_indice_list) == 1:
#                 src_points = self.Data_dict[f'{self.class_indice_list[0]}'][i]
#                 ref_points = self.Data_dict[f'{self.class_indice_list[0]}'][j]
#                 print(f'{self.class_indice_list[0]} - {i} ------- {self.class_indice_list[0]} - {j}')
#             else:
#                 raise ValueError('class length not correct')
#
#         ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
#         src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)
#
#         if len(self.class_indice_list) == 2 and self.class_indice_list[0] == 'chair' and self.class_indice_list[1] == 'bed1':
#             pass
#             # scale = (0.4, 1.0)
#             # raw_points_1, raw_points_2 = normalize_points_two(raw_points_1, raw_points_2, scale)
#             # transform = np.eye(4)
#             # x_max = np.max(raw_points_2, axis=0)[0]
#             # y_max = np.max(raw_points_2, axis=0)[1]
#             # delta_x = x_max - np.min(raw_points_1, axis=0)[0]
#             # delta_y = y_max - np.max(raw_points_1, axis=0)[1]
#             # transform[0:3, 3]=np.array([delta_x, delta_y, 0])
#         # elif All_keys[0] == 'car' and All_keys[1] == 'airplane':
#         #     scale = (0.4, 1.0)
#         #     raw_points_1, raw_points_2 = normalize_points_two(raw_points_1, raw_points_2, scale)
#         #     transform = np.eye(4)
#         #     x_max = np.max(raw_points_2, axis=0)[0]
#         #     y_max = np.max(raw_points_2, axis=0)[1]
#         #     delta_x = x_max - np.min(raw_points_1, axis=0)[0]
#         #     delta_y = y_max - np.max(raw_points_1, axis=0)[1]
#         #     transform[0:3, 3]=np.array([delta_x, delta_y, 0])
#         else:
#             scale = (1., 1.)
#             # ref_points, src_points = normalize_points_two(ref_points, src_points, scale)
#             transform = np.eye(4)
#
#         # src_points = random_sample_points(raw_points_1, self.num_points, normals=None, use_random=self.use_random_sample)
#         # ref_points = random_sample_points(raw_points_2, self.num_points, normals=None, use_random=self.use_random_sample)
#
#         # if self.noise_magnitude is not None:
#             # ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
#             # src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)
#
#         new_data_dict = {
#             # 'raw_points': raw_points.astype(np.float32),
#             'ref_points': ref_points.astype(np.float32),
#             'src_points': src_points.astype(np.float32),
#             'transform': transform.astype(np.float32),
#             # 'label': int(label),
#             'index': int(index),
#         }
#
#
#         return new_data_dict
#
#     def __len__(self):
#         return self.maximal_sample ** 2
#
#
