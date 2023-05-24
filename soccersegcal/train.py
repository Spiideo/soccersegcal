import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
from dataloader import SoccerNetFieldSegmentationDataset, HFlipDataset
from torch.utils.data import DataLoader
import os
import fire
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from torchmetrics import Accuracy, MeanMetric

class LitSoccerFieldSegmentation(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = deeplabv3_resnet50(num_classes=6)
		self.train_loss = MeanMetric()
		self.train_acc = Accuracy("binary")
		self.val_loss = MeanMetric()
		self.val_acc = Accuracy("binary")

	def forward(self, x):
		return self.model(x)['out']

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=1e-5)

	def step(self, batch, loss_metric, acc_metric):
		images, true_masks = batch['image'], batch['segments']
		masks = self.forward(images)
		loss = F.binary_cross_entropy_with_logits(masks.squeeze(1), true_masks)
		loss_metric(loss)
		acc_metric(masks > 0.5, true_masks > 0.5)
		return loss

	def training_step(self, batch, idx):
		loss = self.step(batch, self.train_loss, self.train_acc)
		self.log('train_loss_step', self.train_loss)
		self.log('train_acc_step', self.train_acc)
		return loss

	def on_train_epoch_end(self):
		self.log('train_loss_epoch', self.train_loss)
		self.log('train_acc_epoch', self.train_acc)

	def validation_step(self, batch, idx):
		loss = self.step(batch, self.val_loss, self.val_acc)
		return loss

	def on_validation_epoch_end(self):
		self.log('val_loss', self.val_loss)
		self.log('val_acc', self.val_acc)

def train(datasetpath="data/SoccerNet/calibration-2023", batch_size=4, width=960, limit_batches=None, epochs=1000, logger="mlflow", checkpoint_path=None):
	# Logging
	if logger == "mlflow":
		logger = MLFlowLogger(experiment_name="SoccerFieldSegmentation", log_model=True)
	elif logger == "wandb":
		logger = WandbLogger(project="SoccerFieldSegmentation", log_model=True)
	else:
		raise NotImplementedError
	logger.log_hyperparams({k: v for k, v in locals().items() if k not in ('logger')})

	# Model
	torch.set_float32_matmul_precision('medium')
	model = LitSoccerFieldSegmentation()

	# Data
	train_set = HFlipDataset(SoccerNetFieldSegmentationDataset(width=width, split="train", skip_bad=True, datasetpath=datasetpath))
	val_set = HFlipDataset(SoccerNetFieldSegmentationDataset(width=width, split="valid", skip_bad=True, datasetpath=datasetpath))
	loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count()//2, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

	# Training
	callbacks = [ModelCheckpoint(checkpoint_path, save_last=True, save_top_k=2, monitor='val_acc', mode='max')]
	trainer = pl.Trainer(callbacks=callbacks, logger=logger, precision=16, limit_train_batches=limit_batches, limit_val_batches=limit_batches, max_epochs=epochs)
	trainer.fit(model, train_loader, val_loader, ckpt_path="last")

if __name__ == '__main__':
	fire.Fire(train)