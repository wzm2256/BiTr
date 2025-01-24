import logging
import torch
import numpy as np
import os

class EarlyStopping:
	"""
	Early stops the training if d_loss doesn't improve after a given patience.
	"""
	def __init__(self, patience=100, save_path='.', save_name='', model=None):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 100
			verbose (bool): If True, prints a message for each validation loss improvement.
							Default: False

			model (nn.Module): which model to save.
		"""
		if patience < 0:
			self.best_patience = np.Inf
		else:
			self.best_patience = patience
		self.counter_best = 0
		self.best_step = 0
		self.best_score = -np.Inf
		self.early_stop = False
		self.global_step = -1
		self.path = save_path
		self.save_name = save_name
		self.model = model
		self.logger = logging.getLogger("Early stopping")

	def __call__(self, score, global_step=None):
		if global_step is None:
			self.global_step += 1
		else:
			step = global_step - self.global_step
			assert step > 0, f'global_step {global_step} must be strictly larger than current global_step ({self.global_step})'
			self.global_step = global_step
		if score > self.best_score:
			self.best_score = score
			self.counter = 0
			self.logger.info(f'Best performance {self.best_score : .4f} at epoch: {self.global_step}')
			if self.model is not None:
				self.logger.info('Saving the best model.')
				torch.save(self.model.state_dict(), os.path.join(self.path, f'{self.save_name}_best.pt'))
			self.best_step = self.global_step
			return False, None, None
		else:
			self.counter += step
			if self.counter >= self.best_patience:
				self.logger.info(f'Best performance {self.best_score :.4f} achieved at step: {self.best_step}. Training stop.')
				self.early_stop = True
				return True, self.best_step, self.best_score
			else:
				return False, None, None


if __name__ == '__main__':
	es = EarlyStopping(patience=3)
	logging.basicConfig(level=logging.DEBUG)
	A = torch.randn(20)
	print(A)
	for i in range(20):
		print(i)
		Stop = es(A[i])
		if Stop:
			break

