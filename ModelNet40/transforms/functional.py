import math
import pdb
import random

import numpy as np


def normalize_points(points):
    r"""Normalize point cloud to a unit sphere at origin."""
    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points


def sample_points(points, num_samples, normals=None):
    r"""Sample the first K points."""
    points = points[:num_samples]
    if normals is not None:
        normals = normals[:num_samples]
        return points, normals
    else:
        return points

def normalize_points_two(raw_points_1, raw_points_2, scale):
    raw_points_1 = raw_points_1 - raw_points_1.mean(axis=0)
    raw_points_1 = raw_points_1 / np.max(np.linalg.norm(raw_points_1, axis=1)) * scale[0]

    raw_points_2 = raw_points_2 - raw_points_2.mean(axis=0)
    raw_points_2 = raw_points_2 / np.max(np.linalg.norm(raw_points_2, axis=1)) * scale[1]

    return raw_points_1, raw_points_2

def random_sample_points(points, num_samples_, normals=None, color=None, use_random=True):
    r"""Randomly sample points."""
    num_points = points.shape[0]
    # pdb.set_trace()
    if use_random:
        sel_indices = np.random.permutation(num_points)
    else:
        rng = np.random.RandomState(0)
        sel_indices = rng.permutation(num_points)
    if num_samples_ < 0:
        num_samples = num_points
    else:
        num_samples = num_samples_

    if num_points >= num_samples:
        sel_indices = sel_indices[:num_samples]
    else:
        raise ValueError('Requiring too many points!')
    points = points[sel_indices]
    if color is not None:
        colors = color[sel_indices]
        if normals is not None:
            normals = normals[sel_indices]
            return points, normals, colors
        else:
            return points, colors
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points


def random_scale_shift_points(points, low=2.0 / 3.0, high=3.0 / 2.0, shift=0.2, normals=None):
    r"""Randomly scale and shift point cloud."""
    scale = np.random.uniform(low=low, high=high, size=(1, 3))
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points * scale + bias
    if normals is not None:
        normals = normals * scale
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return points, normals
    else:
        return points


def random_rotate_points_along_up_axis(points, normals=None):
    r"""Randomly rotate point cloud along z-axis."""
    theta = np.random.rand() * 2.0 * math.pi
    # fmt: off
    rotation_t = np.array([
        [math.cos(theta), math.sin(theta), 0],
        [-math.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ])
    # fmt: on
    points = np.matmul(points, rotation_t)
    if normals is not None:
        normals = np.matmul(normals, rotation_t)
        return points, normals
    else:
        return points


def random_rescale_points(points, low=0.8, high=1.2):
    r"""Randomly rescale point cloud."""
    scale = random.uniform(low, high)
    points = points * scale
    return points


def random_jitter_points(points, scale, noise_magnitude=0.05):
    r"""Randomly jitter point cloud."""
    noises = np.clip(np.random.normal(scale=scale, size=points.shape), a_min=-noise_magnitude, a_max=noise_magnitude)
    points = points + noises
    return points


def random_shuffle_points(points, normals=None):
    r"""Randomly permute point cloud."""
    indices = np.random.permutation(points.shape[0])
    points = points[indices]
    if normals is not None:
        normals = normals[indices]
        return points, normals
    else:
        return points


def random_dropout_points(points, max_p):
    r"""Randomly dropout point cloud proposed in PointNet++."""
    num_points = points.shape[0]
    p = np.random.rand(num_points) * max_p
    masks = np.random.rand(num_points) < p
    points[masks] = points[0]
    return points


def random_jitter_features(features, mu=0, sigma=0.01):
    r"""Randomly jitter features in the original implementation of FCGF."""
    if random.random() < 0.95:
        features = features + np.random.normal(mu, sigma, features.shape).astype(np.float32)
    return features


def random_sample_plane_bak():
    r"""Random sample a plane passing the origin and return its normal."""
    phi = np.random.uniform(0.0, 2 * np.pi)  # longitude
    theta = np.random.uniform(0.0, np.pi)  # latitude

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    normal = np.asarray([x, y, z])
    return normal

def random_sample_plane(use_random_plane=True):
    r"""Random sample a plane passing the origin and return its normal."""
    if use_random_plane:
        a = np.random.randn(3) * 100
    else:
        rng = np.random.RandomState(0)
        a = rng.rand(3) * 100
    norm = np.linalg.norm(a) + 1e-8
    return a / norm

def random_crop_point_cloud_with_plane(points, p_normal=None, keep_ratio=0.7, normals=None):
    r"""Random crop a point cloud with a plane and keep num_samples points."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if p_normal is None:
        p_normal = random_sample_plane()  # (3,)
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples]  # select the largest K points
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points

def random_seperate_point_cloud(points, p_normal=None, keep_ratio=0.7, use_random_plane=True):
    r"""Random crop a point cloud with a plane and keep num_samples points."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if p_normal is None:
        p_normal = random_sample_plane(use_random_plane=use_random_plane)  # (3,)
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples]  # select the largest K points
    nosel_indices = np.argsort(-distances)[num_samples:]  # select the largest K points
    # pdb.set_trace()
    points_select = points[sel_indices]
    points_left = points[nosel_indices]
    return points_select, points_left


def random_sample_viewpoint(limit=500):
    r"""Randomly sample observing point from 8 directions."""
    return np.random.rand(3) + np.array([limit, limit, limit]) * np.random.choice([1.0, -1.0], size=3)


def random_crop_point_cloud_with_point(points, viewpoint=None, keep_ratio=0.7, normals=None):
    r"""Random crop point cloud from the observing point."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if viewpoint is None:
        viewpoint = random_sample_viewpoint()
    distances = np.linalg.norm(viewpoint - points, axis=1)
    sel_indices = np.argsort(distances)[:num_samples]
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points
