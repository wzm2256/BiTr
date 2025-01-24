import os
import pdb

from torch.utils.tensorboard import SummaryWriter
import dgl
import numpy as np
import torch
from torch import optim
import logging
#############
from Model.BiNet import BiSE3TransformerDown as BiSE3Transformer

from early_stopping import EarlyStopping

import utils
from Model.fiber_new import Fiber, fiber_merge, generate_fiber
import einops

try:
	import icp
except:
	print('icp not imported.')

import utils_sinkhorn


def test_one_pc(g1, g2, model_all, FLAGS, r_gt, t_gt, vis=0, i=0):
	with torch.no_grad():
		g1 = g1.to('cuda')
		g2 = g2.to('cuda')


		if FLAGS.icp == 'only':
			r_isotropic, t_isotropic, r, t = icp.icp_batch_graph(g1, g2, r_gt, t_gt)
			return r_isotropic.mean(), t_isotropic.mean(), torch.tensor(0.), r.to('cuda'), t.to('cuda')
		if FLAGS.icp == 'only_sinkhorn':
			r_isotropic, t_isotropic, r, t = utils_sinkhorn.OT_compute(g1, g2, r_gt, t_gt)
			return r_isotropic.mean(), t_isotropic.mean(), torch.tensor(0.), r.to('cuda'), t.to('cuda')

		g_batch = dgl.batch([g1, g2])

		feature_dict_batch = {}
		if FLAGS.return_normals:
			feature_dict_batch[(1, 0)] = torch.einsum('ji,bcj->bci', torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device='cuda'),
			                                    g_batch.ndata['norm_vec'].unsqueeze(1))
		if FLAGS.get_color:
			feature_dict_batch[(0, 0)] = g_batch.ndata['c'].unsqueeze(-1)
		else:
			feature_dict_batch[(0, 0)] = g_batch.ndata['c'].reshape((g_batch.ndata['c'].shape[0], -1, 1)) * 0.5


		r, t, g1_select, g2_select, feature, r_vector = model_all(feature_dict=feature_dict_batch, g=g_batch)


		if FLAGS.complete_match == 1:
			assert FLAGS.get_color == False
			assert FLAGS.return_normals == False
			g_batch_id = dgl.batch([g1, g1])
			r_i, t_i, g1_select_i, g2_select_i, feature_i, r_vector_i = model_all(feature_dict=feature_dict_batch, g=g_batch_id)
			trans_ori = torch.eye(4).to('cuda')
			trans_ori[:3,:3] = r
			trans_ori[:3, 3] = t.squeeze(0)

			trans_i = torch.eye(4).to('cuda')
			trans_i[:3,:3] = r_i
			trans_i[:3, 3] = t_i.squeeze(0)

			trans_final = trans_ori @ trans_i
			r = trans_final[:3, :3].unsqueeze(0)
			t = trans_final[:3, 3].unsqueeze(0)

		Loss = utils.loss_4(r, r_gt, t, t_gt, coe=1.0)

		if FLAGS.icp == 'finetune':
			r_isotropic, t_isotropic, r, t = icp.icp_batch_graph(g1, g2, r_gt, t_gt, r=r, t=t)
			r = r.to('cuda')
			t = t.to('cuda')
		if FLAGS.icp == 'sinkhorn':
			r_isotropic, t_isotropic, r, t = utils_sinkhorn.OT_compute(g1, g2, r_gt, t_gt, r=r, t=t)
			r = r.to('cuda')
			t = t.to('cuda')
		else:
			r_isotropic, t_isotropic, euler_error = utils.compute_metrics(r, t, r_gt, t_gt)
		FLAGS.test_logger.debug(f'Test {i}: Rot error: {r_isotropic.mean().item() :.4f} trans error: {t_isotropic.mean().item() :.4f}')



		if vis == 1:
			pdb.set_trace()
			g1_list = dgl.unbatch(g1)
			g2_list = dgl.unbatch(g2)

			g2_select_list = dgl.unbatch(g2_select)
			g1_select_list = dgl.unbatch(g1_select)
			for j in range(len(g1_list)):
				np.save(f'Vis/gt_{i * FLAGS.batch_size + j}.npy', (g1_list[j].ndata['pos'] @ r_gt[j].T + t_gt[j].unsqueeze(0)).cpu().numpy())
				np.save(f'Vis/Ref_{i * FLAGS.batch_size + j}.npy', g2_list[j].ndata['pos'].cpu().numpy())
				np.save(f'Vis/Source_{i * FLAGS.batch_size + j}.npy', g1_list[j].ndata['pos'].cpu().numpy())
				np.save(f'Vis/Ref_select_{i * FLAGS.batch_size + j}.npy', g2_select_list[j].ndata['pos'].cpu().numpy())
				np.save(f'Vis/Src_select_{i * FLAGS.batch_size + j}.npy', g1_select_list[j].ndata['pos'].cpu().numpy())

				predict_g1 = g1_list[j].ndata['pos'].to('cuda') @ r[j].T + t[j].unsqueeze(0)
				predict_select_g1 = g1_select_list[j].ndata['pos'].to('cuda') @ r[j].T + t[j].unsqueeze(0)

				np.save(f'Vis/Predict_{i * FLAGS.batch_size + j}.npy', predict_g1.cpu().numpy())
				np.save(f'Vis/Predict_select_{i * FLAGS.batch_size + j}.npy', predict_select_g1.cpu().numpy())

				if FLAGS.get_color:
					np.save(f'Vis/Ref_color_{i * FLAGS.batch_size + j}.npy', g2_list[j].ndata['c'].cpu().numpy())
					np.save(f'Vis/Source_color_{i * FLAGS.batch_size + j}.npy', g1_list[j].ndata['c'].cpu().numpy())
		else:
			FLAGS.writer.add_scalar('Test/r', r_isotropic.mean().item(), i)
			FLAGS.writer.add_scalar('Test/t', t_isotropic.mean().item(), i)

		return r_isotropic.mean(), t_isotropic.mean(), Loss.mean(), r, t, euler_error

def test_epoch(epoch, dataloader, FLAGS, vis=0, model_all=None, rotation=False, flip=False, scale=False, name=None):
	Record = {
		'R': [],
		't': [],
		'L': [],
		'W': [],
		'Euler' :[]
	}

	record_txt = open(f'{name}.txt', 'a')
	for re in range(FLAGS.test_repeat):
		for s, (g2_origin, g1, transform) in enumerate(dataloader):
			r_gt_origin = transform[:, 0:3, 0:3].to('cuda')
			t_gt_origin = transform[:, 0:3, 3].to('cuda')

			if rotation:
				###############
				from scipy.spatial.transform import Rotation as R
				r2 = R.random().as_matrix().astype(np.float32)
				t2 = np.random.rand(1, 3).astype(np.float32)
				g2_origin.ndata['pos'] = g2_origin.ndata['pos'].to('cpu') @ torch.tensor(r2).T + t2
				r_gt_origin = (torch.tensor(r2).to('cuda') @ r_gt_origin[0]).unsqueeze(0)


				assert r_gt_origin.shape[0] == 1, 'Only support batch 1 rotation visualization.'
				assert FLAGS.on_the_fly == 'simple', 'Only support simple basis computation'
				# print('Rotate g2....')
				FLAGS.test_logger.info('Rotate g2....')
				Rot_list, trans_list = utils.get_continuous_se3(10)
				if 1 == 1:
					i = 2
					j = 7

					g2 = g2_origin.clone().to('cpu')
					g2.ndata['pos'] = g2_origin.ndata['pos'].to('cpu') @ Rot_list[i].T.to('cpu') + trans_list[i].unsqueeze(0).to('cpu')
					if FLAGS.return_normals:
						g2.ndata['norm_vec'] = g2_origin.ndata['norm_vec'] @ Rot_list[i].T.to('cpu')

					#############
					r_gt = Rot_list[i] @ r_gt_origin
					t_gt = (trans_list[i] + Rot_list[i] @ t_gt_origin[0:1].squeeze(0)).unsqueeze(0)
					r_isotropic, t_isotropic, Loss, r, t = test_one_pc(g1, g2, model_all, FLAGS, r_gt, t_gt, vis=vis, i=2)
					trans_rot = torch.eye(4).to('cuda')
					trans_rot[:3, :3] = r.clone()
					trans_rot[:3, 3:] = t.T.clone()

					trans_r2 = torch.eye(4).to('cuda')
					trans_r2[:3, :3] = Rot_list[i].clone()
					trans_r2[:3, 3] = trans_list[i].clone()
					record_txt.write(f'{r_isotropic.item():.2f}\t{t_isotropic.item():.2f}\t')

			if flip:
				assert r_gt_origin.shape[0] == 1, 'Only support batch 1 flip visualization.'
				assert FLAGS.on_the_fly == 'simple', 'Only support simple basis computation'
				FLAGS.test_logger.info('Swap x and y ....')
				t_gt = -(r_gt_origin[0].T @ t_gt_origin.T).T
				r_gt = r_gt_origin.mT
				r_isotropic, t_isotropic, Loss, r, t  = test_one_pc(g2_origin, g1, model_all, FLAGS, r_gt, t_gt, vis=vis,
				                                             i=1)
				record_txt.write(f'{r_isotropic.item():.2f}\t{t_isotropic.item():.2f}\t')
				trans_flip = torch.eye(4).to('cuda')
				trans_flip[:3, :3] = r.clone()
				trans_flip[:3, 3:] = t.T.clone()

			if scale:
				assert r_gt_origin.shape[0] == 1, 'Only support batch 1 flip visualization.'
				assert FLAGS.on_the_fly == 'simple', 'Only support simple basis computation'
				Scale_change_list = [4]
				FLAGS.test_logger.info(f'Scale change {Scale_change_list}....')
				for scale_index, scale in enumerate(Scale_change_list):
					g2_scale = g2_origin.clone().to('cpu')
					g2_scale.ndata['pos'] = g2_origin.ndata['pos'] * scale
					g1_scale = g1.clone().to('cpu')
					g1_scale.ndata['pos'] = g1.ndata['pos'] * scale
					t_gt = t_gt_origin * scale
					r_isotropic, t_isotropic, Loss, r, t = test_one_pc(g1_scale, g2_scale, model_all, FLAGS, r_gt_origin, t_gt, vis=vis, i=s + 3 + scale_index)
					record_txt.write(f'{r_isotropic.item():.2f}\t{t_isotropic.item():.2f}\n')
					trans_s = torch.eye(4).to('cuda')
					trans_s[:3, :3] = r.clone()
					trans_s[:3, 3:] = t.T.clone()
				print('------------')


			g2 = g2_origin
			r_gt = r_gt_origin
			t_gt = t_gt_origin
			r_isotropic, t_isotropic, Loss, r, t, euler_error = test_one_pc(g1, g2, model_all, FLAGS, r_gt, t_gt, vis=vis, i=epoch*len(dataloader)+s)

			Record['R'].append(r_isotropic)
			Record['t'].append(t_isotropic)
			Record['L'].append(Loss)
			Record['W'].append(t_gt_origin.shape[0])
			Record['Euler'].append(euler_error)
			if vis == 1 and (rotation or flip):
				return


	R_err_mean = (torch.tensor(Record['W']) * torch.stack(Record['R']).cpu()).sum() / torch.tensor(Record['W']).sum()
	t_err_mean = (torch.tensor(Record['W']) * torch.stack(Record['t']).cpu()).sum() / torch.tensor(Record['W']).sum()
	L_err_mean = (torch.tensor(Record['W']) * torch.stack(Record['L']).cpu()).sum() / torch.tensor(Record['W']).sum()
	if FLAGS.euler:
		Euler_err_mean = (torch.tensor(Record['W']).unsqueeze(1) * torch.tensor(np.stack(Record['Euler'])).squeeze(1)).sum(0) / torch.tensor(Record['W']).sum()

	FLAGS.writer.add_scalar('Test_epoch/R', R_err_mean.item(), epoch)
	FLAGS.writer.add_scalar('Test_epoch/t', t_err_mean.item(), epoch)
	FLAGS.writer.add_scalar('Test_epoch/L', L_err_mean.item(), epoch)

	FLAGS.test_logger.info(f'Summary: Test: Rot error: {R_err_mean.item() :.4f} trans error: {t_err_mean.item() :.4f}')
	if FLAGS.euler:
		FLAGS.test_logger.info(f'Summary: Test: Euler error: {Euler_err_mean} ')


	return R_err_mean, t_err_mean, L_err_mean

def train_epoch(epoch, dataloader, optimizer, FLAGS, model_all=None):

	for i, (g2, g1, transform) in enumerate(dataloader):
		r_gt = transform[:, 0:3, 0:3].to('cuda')
		t_gt = transform[:, 0:3, 3].to('cuda')

		g1 = g1.to(FLAGS.device)
		g2 = g2.to(FLAGS.device)
		g_batch = dgl.batch([g1, g2])

		feature_dict_batch = {}
		if FLAGS.return_normals:
			feature_dict_batch[(1, 0)] = torch.einsum('ji,bcj->bci', torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device='cuda'),
			                                    g_batch.ndata['norm_vec'].unsqueeze(1))
		if FLAGS.get_color:
			feature_dict_batch[(0, 0)] = g_batch.ndata['c'].unsqueeze(-1)
		else:
			feature_dict_batch[(0, 0)] = g_batch.ndata['c'].reshape((g_batch.ndata['c'].shape[0], -1, 1)) * 0.5

		r, t, g1_key_graph, g2_key_graph, feature, r_vec = model_all(feature_dict=feature_dict_batch, g=g_batch)

		g1_key_pos = einops.rearrange(g1_key_graph.ndata['pos'], '(b p) d -> b p d', p=FLAGS.num_key_points)
		g2_key_pos = einops.rearrange(g2_key_graph.ndata['pos'], '(b p) d -> b p d', p=FLAGS.num_key_points)
		kp_nn_dist = ((g1_key_pos.unsqueeze(1) - g1_key_pos.unsqueeze(2)) ** 2).sum(-1).max()
		nn_dist = torch.clamp((g1_key_pos.unsqueeze(1) - g1_key_pos.unsqueeze(2)).abs().sum(-1), max=FLAGS.key_dist_thre).mean() + \
		          torch.clamp((g2_key_pos.unsqueeze(1) - g2_key_pos.unsqueeze(2)).abs().sum(-1), max=FLAGS.key_dist_thre).mean()
		NN_reg = - nn_dist * FLAGS.key_dist_coe


		Loss = utils.loss_4(r, r_gt, t, t_gt, coe=1.0)
		Loss = Loss + NN_reg

		r_isotropic, t_isotropic, _ = utils.compute_metrics(r, t, r_gt, t_gt)

		FLAGS.train_logger.debug(f'NN-reg: {NN_reg.item()}')
		FLAGS.train_logger.info(f'Epoch {epoch} Step {i} Loss: {Loss.item() :.4f} Rot error: {r_isotropic.mean().item():.4f} trans error: {t_isotropic.mean().item():.4f}')
		if i % 10 == 0:
			FLAGS.writer.add_scalar('Train/Loss', Loss, epoch * len(dataloader) + i)
			FLAGS.writer.add_scalar('Train/R_error', r_isotropic.mean().item(), epoch * len(dataloader) + i)
			FLAGS.writer.add_scalar('Train/t_error', t_isotropic.mean().item(), epoch * len(dataloader) + i)
			FLAGS.writer.add_scalar('Train/NN_reg', NN_reg.mean().item(), epoch * len(dataloader) + i)
			FLAGS.writer.add_scalar('Train/kp_nn_dist', kp_nn_dist.mean().item(), epoch * len(dataloader) + i)


		optimizer.zero_grad()
		Loss.backward()

		torch.nn.utils.clip_grad_value_(model_all.parameters(), 1)

		optimizer.step()


def main(FLAGS):
	# Prepare data
	train_dataset = GraphDataSet(
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

	test_dataset = GraphDataSet(
		subset='test',
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

	train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=False, num_workers=FLAGS.num_workers)
	test_loader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, drop_last=False, num_workers=FLAGS.num_workers)

	# Construct fiber for feature network
	max_degree_side = (np.inf, 0)
	fiber_hidden = generate_fiber(FLAGS.feature_max_deg, max_degree_side, FLAGS.feature_num_channels) # degree (0, 0) and (0, 1)
	fiber_out = Fiber(fiber={(1, 0): FLAGS.feature_num_channels, (0, 0): FLAGS.feature_num_channels})
	fiber_in_dict = {}
	if FLAGS.return_normals:
		fiber_in_dict[(1, 0)] = 1
	if FLAGS.get_color:
		fiber_in_dict[(0, 0)] = 3
	else:
		fiber_in_dict[(0, 0)] = 1
	fiber_in = Fiber(fiber=fiber_in_dict)

	fiber_feature = {'in': fiber_in, 'mid': Fiber(fiber=fiber_hidden), 'out': fiber_out}

	# ------------------- two side
	max_degree_side = (1, 1)
	fiber_binet_in = fiber_merge(fiber_feature['out'], fiber_feature['out'], type=FLAGS.merge_type)
	fiber_binet_hidden = generate_fiber(FLAGS.max_deg, max_degree_side, FLAGS.num_channels)
	fiber_binet = {'in': fiber_binet_in,
                    'mid': Fiber(fiber=fiber_binet_hidden),
                    'out': Fiber(fiber={(1, 1): FLAGS.predict_head, (0, 1): FLAGS.predict_head,
	                                    (1, 0): FLAGS.predict_head, (0, 0): 4 * FLAGS.predict_head})}


	model_all = BiSE3Transformer(fiber_feature, FLAGS.feature_num_layers, fiber_binet, FLAGS.num_layers,
                div=FLAGS.div, n_heads=FLAGS.head, max_deg_total=FLAGS.max_deg, max_degree_side=max_degree_side,
                feature_max_deg_total=FLAGS.feature_max_deg, merge_type=FLAGS.merge_type,
	            num_head=FLAGS.predict_head, re_convert_k=FLAGS.re_convert_k, num_key_points=FLAGS.num_key_points,
	            key_mean_num=FLAGS.key_mean_num, key_global_merge_coe=FLAGS.key_global_merge_coe,
	                             )



	model_all.to(FLAGS.device)
	optimizer = optim.Adam(model_all.parameters(), lr=FLAGS.lr)

	save_path_all = os.path.join(FLAGS.save_dir, FLAGS.exp_name)

	if FLAGS.only_test == 1:
		if FLAGS.num_epochs == -1:
			tar_loaded = torch.load(save_path_all + '_' + 'best' + '.pt')
		elif FLAGS.num_epochs == -2:
			tar_loaded = torch.load(save_path_all + '_' + 'last_' + '.pt')
		elif FLAGS.num_epochs == -3:
			pass
		else:
			tar_loaded = torch.load(save_path_all + '_' + str(FLAGS.num_epochs) + '_.pt')

		if FLAGS.num_epochs != -3:
			if 'model_state_dict' in tar_loaded:
				model_all.load_state_dict(tar_loaded['model_state_dict'], strict=False)
			else:
				model_all.load_state_dict(tar_loaded, strict=False)
		test_epoch(0, test_loader, FLAGS, vis=0, model_all=model_all, rotation=False, flip=False, scale=False, name=FLAGS.exp_name)
		return

	elif FLAGS.load_epoch >= 0:
		if FLAGS.load_epoch == 0:
			tar_loaded = torch.load(save_path_all + '_' + 'last_' + '.pt')
		else:
			tar_loaded = torch.load(save_path_all + '_' + str(FLAGS.load_epoch) + '_.pt')
		model_all.load_state_dict(tar_loaded['model_state_dict'])
		optimizer.load_state_dict(tar_loaded['optimizer_state_dict'])
		epoch_start = tar_loaded['epoch']
	else:
		epoch_start = 0


	es = EarlyStopping(patience=-1, save_path=FLAGS.save_dir, save_name=FLAGS.exp_name, model=model_all)
	# Run training
	FLAGS.train_logger.info('Begin training')
	for epoch in range(epoch_start, FLAGS.num_epochs + epoch_start):

		train_epoch(epoch, train_loader, optimizer, FLAGS, model_all=model_all)
		if epoch % 5 == 0:
			R_err_mean, t_err_mean, L_err_mean = test_epoch(epoch, test_loader, FLAGS, vis=0, model_all=model_all)
			stop, best_step, best_score = es(-R_err_mean, global_step=epoch)

			if FLAGS.early_terminate == 1:
				if epoch > 500 and R_err_mean > 80:
					break

			if stop:
				break
			if FLAGS.save_model == 1 and epoch % FLAGS.save_freq == 0:
				torch.save({ 'epoch': epoch, 'model_state_dict': model_all.state_dict(),
				             'optimizer_state_dict': optimizer.state_dict()}, save_path_all + '_' + str(epoch) + '_.pt')
				FLAGS.train_logger.info(f"Saved: {save_path_all}")

	torch.save({ 'epoch': epoch, 'model_state_dict': model_all.state_dict(),
	             'optimizer_state_dict': optimizer.state_dict()}, save_path_all + '_' + 'last' + '_.pt')
	FLAGS.train_logger.info(f"Last model saved: {save_path_all}")

if __name__ == '__main__':
	import args

	FLAGS = args.get_flags()
	os.makedirs(FLAGS.save_dir, exist_ok=True)

	tf_log = os.path.join('LOG', FLAGS.exp_name)
	if not os.path.isdir(tf_log):
		os.makedirs(tf_log)
	FLAGS.writer = SummaryWriter(tf_log)

	log_file = os.path.join(tf_log, 'log.txt')

	handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	logging.basicConfig(level=logging.INFO, handlers=handlers)
	FLAGS.train_logger = logging.getLogger("training")
	FLAGS.test_logger = logging.getLogger("test")
	FLAGS.train_logger.info(FLAGS)


	if FLAGS.data_name == 'BB':
		from ModelNet40.BBGraphDataset import BBGraph as GraphDataSet
		FLAGS.train_logger.info('Loading BB dataset.....')
	elif FLAGS.data_name == 'ShapeNet':
		from ModelNet40.ModelNetGraphDataset import ModelnetGraph as GraphDataSet
		FLAGS.train_logger.info('Loading ModelNet dataset.....')
	elif FLAGS.data_name == 'bunny':
		from ModelNet40.BunnyGraphDataset import BunnyGraph as GraphDataSet
	elif FLAGS.data_name == 'Indoor':
		from ModelNet40.IndoorGraphDataset import MatchGraph as GraphDataSet
		FLAGS.train_logger.info('Loading Indoor dataset.....')
	elif FLAGS.data_name == 'ASL':
		from ModelNet40.ASLGraphDataset import MatchGraph as GraphDataSet
		FLAGS.train_logger.info('Loading asl dataset.....')
	else:
		raise NotImplementedError

	main(FLAGS)