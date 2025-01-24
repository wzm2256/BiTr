try:
	import open3d as o3d
except:
	print('Warnining open3d is not loaded. No visualization.')
import torch
import numpy as np
import math
import einops

try:
	import escnn.group as group
except:
	pass
from scipy.spatial.transform import Rotation as R

def get_continuous_se3(N):
	alpha = np.linspace(0, np.pi * 2, N)
	beta = np.linspace(0, np.pi, N)
	theta = np.linspace(0, np.pi * 2, N)

	Rot_list = []
	trans_list = []
	t_old = torch.randn(3).to('cuda')
	t_old = t_old / t_old.norm()

	for i in range(N):
		r = R.from_quat([np.sin(theta[i] / 2) * np.sin(alpha[i]) * np.cos(beta[i]),
						 np.sin(theta[i] / 2) * np.sin(alpha[i]) * np.sin(beta[i]),
						 np.sin(theta[i] / 2) * np.cos(alpha[i]), np.cos(theta[i] / 2)])
		Rot_list.append(torch.tensor(r.as_matrix(), dtype=torch.float).to('cuda'))
		trans_list.append(t_old * (3 * i / N))

	return Rot_list, trans_list


def loss_4(r, r_gt, t, t_gt, coe=10.):

	r_loss = ((einops.einsum(r_gt, r.mT, 'b c d, b d e -> b c e') -
			   torch.eye(3, device='cuda').unsqueeze(0)) ** 2).mean()
	t_loss = ((t_gt - t) ** 2).mean()
	Loss =  r_loss + coe * t_loss
	return Loss


def isotropic_R_error(r1, r2):
	'''
	Calculate isotropic rotation degree error between r1 and r2.
	:param r1: shape=(B, 3, 3), pred
	:param r2: shape=(B, 3, 3), gt
	:return:
	'''
	r2_inv = r2.permute(0, 2, 1).contiguous()
	r1r2 = torch.matmul(r2_inv, r1)
	############
	# tr_old = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
	############
	tr = torch.vmap(torch.trace)(r1r2)
	############
	rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
	degrees = rads / math.pi * 180
	return degrees



def matrix2euler(mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
    """Converts rotation matrix to euler angles

    Args:
        mats: (B, 3, 3) containing the B rotation matricecs
        seq: Sequence of euler rotations (default: 'zyx')
        degrees (bool): If true (default), will return in degrees instead of radians

    Returns:

    """

    eulers = []
    for i in range(mats.shape[0]):
        r = R.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return np.stack(eulers)


def compute_metrics(R, t, gtR, gtt, euler=False):
	cur_r_isotropic = isotropic_R_error(R, gtR)
	cur_t_isotropic = (t - gtt).norm(dim=-1)

	r_gt_euler_deg = matrix2euler(gtR[:, :3, :3].detach().cpu().numpy(), seq='zxy')
	r_pred_euler_deg = matrix2euler(R[:, :3, :3].detach().cpu().numpy(), seq='zxy')
	euler_error =  np.abs(r_gt_euler_deg - r_pred_euler_deg)
	if euler:
		return cur_r_isotropic, cur_t_isotropic, euler_error
	return cur_r_isotropic, cur_t_isotropic, 0.0


def view_with_direction(LandScape, point_size=8., parameters=None, savename=None):
	vis = o3d.visualization.Visualizer()
	vis.create_window()

	if isinstance(LandScape, list):
		for i in LandScape:
			vis.add_geometry(i)
	else:
		vis.add_geometry(LandScape)

	if parameters is not None:
		ctr = vis.get_view_control()
		ctr.convert_from_pinhole_camera_parameters(parameters)

	render = vis.get_render_option()
	render.point_size = point_size

	vis.run()
	if savename is not None:
		vis.capture_screen_image(savename)



def visualize_3d_3(A, viewpoint=None, point_size=1., savename=None):
	template = o3d.geometry.PointCloud()
	template.points = o3d.utility.Vector3dVector(A[0])

	sample = o3d.geometry.PointCloud()
	sample.points = o3d.utility.Vector3dVector(A[1])

	sample2 = o3d.geometry.PointCloud()
	sample2.points = o3d.utility.Vector3dVector(A[2])

	template.paint_uniform_color([1, 0., 0.])
	sample.paint_uniform_color([0, 0.651, 0.929])
	sample2.paint_uniform_color([1, 0.706, 0])

	if viewpoint != None:
		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
	else:
		parameters = None

	if savename is not None:
		current_savename = savename + '_vis3.png'
	else:
		current_savename = None

	view_with_direction([template, sample, sample2], parameters=parameters, point_size=point_size, savename=current_savename)



def visualize_3d_2(A, viewpoint=None, point_size=1., savename=None):
	template = o3d.geometry.PointCloud()
	template.points = o3d.utility.Vector3dVector(A[0])

	sample2 = o3d.geometry.PointCloud()
	sample2.points = o3d.utility.Vector3dVector(A[1])
	sample2.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0., 0., 1.]]), [A[1].shape[0], 1]))

	template.paint_uniform_color([1, 0, 0])
	sample2.paint_uniform_color([0, 0, 1.])

	if viewpoint != None:
		parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
	else:
		parameters = None

	if savename is not None:
		my_savename = savename + '_vis2.png'
	else:
		my_savename = None
	view_with_direction([template, sample2], parameters=parameters, point_size=point_size, savename=my_savename)



def vis_PC(A, savename=None, viewpoint=None , point_size=1.):
	if len(A) == 2:
		visualize_3d_2(A, savename=savename, viewpoint=viewpoint, point_size=point_size)
	else:
		visualize_3d_3(A, savename=savename, viewpoint=viewpoint, point_size=point_size)
