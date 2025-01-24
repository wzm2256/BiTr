import argparse
import pdb

import torch


def int2bool(x):
	# pdb.set_trace()
	return str(x) == '1'

def str_None(x):
	if x == 'None':
		return None
	else:
		return x

def float_None(x):
	if x == 'None':
		return None
	else:
		return float(x)

def get_flags():
	parser = argparse.ArgumentParser()

	## Experiment setting.
	parser.add_argument('--exp_name', type=str, default='Exp')

	## Dataset
	parser.add_argument('--data_name', type=str, default='ModelNet', help='dataset name')
	parser.add_argument('--num_points', type=int, default=1024, help='Number of points')
	parser.add_argument('--noise_magnitude', type=float_None, default=0., help='Magnitude of added noise')
	parser.add_argument('--rotation_magnitude', type=float, default=0., help='Magnitude of added rotation')
	parser.add_argument('--translation_magnitude', type=float, default=0., help='Magnitude of added translation')
	parser.add_argument('--keep_ratio', type=float, default=0.7, help='Ratio of kept points')
	parser.add_argument('--class_indices', type=str_None, default='None', help='The class used for training')
	parser.add_argument('--return_normals', type=int2bool, default='0', help='Use normals or not')
	parser.add_argument('--k_neighbor', type=int, default=24, help='Number of neighbors for knn')
	parser.add_argument('--get_color', type=int2bool, default='0', help='Use color or not')
	parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
	parser.add_argument('--num_workers', type=int, default=0, help="Number of data loader workers")

	## Network setting
	parser.add_argument('--num_channels', type=int, default=4, help="Number of channels in middle layers")
	parser.add_argument('--predict_head', type=int, default=2, help="Number of predict head")
	parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in Binet")
	parser.add_argument('--feature_num_layers', type=int, default=2, help="Number of layers in feature extractor")
	parser.add_argument('--feature_max_deg', type=int, default=1, help='Max degree in feature extractor')
	parser.add_argument('--feature_num_channels', type=int, default=4, help="Number of channels in feature extractor")
	parser.add_argument('--div', type=int, default=2, help="Low dimensional embedding fraction")
	parser.add_argument('--head', type=int, default=2, help="Number of attention heads")
	parser.add_argument('--max_deg', type=int, default=2, help='Max degree')
	parser.add_argument('--re_convert_k', type=int, default=24, help='')
	parser.add_argument('--num_key_points', type=int, default=32, help='')
	parser.add_argument('--key_mean_num', type=int, default=24, help='')
	parser.add_argument('--merge_type', type=str, default='11', help='')
	parser.add_argument('--key_global_merge_coe', type=int, default=4, help='')



	parser.add_argument('--test_repeat', type=int, default=1, help='')
	parser.add_argument('--icp', type=str, default='none', help='only|only_sinkhorn|finetune|sinkhorn.')


	# Training
	parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
	parser.add_argument('--num_epochs', type=int, default=5000, help="Number of epochs")
	parser.add_argument('--key_dist_coe', type=float, default=0.1, help='')
	parser.add_argument('--key_dist_thre', type=float, default=0.1, help='')
	parser.add_argument('--save_model', type=int, default=0, help="")
	parser.add_argument('--save_freq', type=int, default=2000, help="")
	parser.add_argument('--save_dir', type=str, default="saved_model", help="Directory name to save models")
	parser.add_argument('--only_test', type=int, default=0, help="Path to model to restore")
	parser.add_argument('--early_terminate', type=int, default=0, help='')
	parser.add_argument('--load_epoch', type=int, default=-4, help='')


	# parser.add_argument('--seed', type=int, default=1992)

	# others
	parser.add_argument('--complete_match', type=int, default=0, help='Test complete matching property (Prop C.11).')
	parser.add_argument('--euler', type=int2bool, default='0', help='')


	FLAGS, UNPARSED_ARGV = parser.parse_known_args()

	# FLAGS = parser.parse_args()

	# Automatically choose GPU if available
	if torch.cuda.is_available():
		FLAGS.device = torch.device('cuda')
	else:
		FLAGS.device = torch.device('cpu')

	return FLAGS
