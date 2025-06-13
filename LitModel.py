import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import random
from optimizers import *
import models
from retriever import *
from custom_metrics import *

from torchmetrics.classification.accuracy import Accuracy



class LitModel(pl.LightningModule):
    """Transfer Learning
    """
    def __init__(self,
                 backbone: str = 'srnet',
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 eps: float = 1e-7,
                 lr_scheduler_name: str = 'lrdrop',
                 optimizer_name: str = 'adamax',
                 num_workers: int = 6,
                 epochs: int = 50,
                 gpus: str = '0',
                 weight_decay: float = 1e-5,
                 seed: str ='',
                 loss_f: str='cross_entropy',
                 warmup: int = 0,
                 imsize: int = 512
                 ,**kwargs) -> None:

        super().__init__()
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_name = optimizer_name
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.seed = seed
        self.loss_f = loss_f
        self.warmup = warmup
        self.imsize = imsize

        self.save_hyperparameters()

        if self.loss_f == 'mse':
            self.train_metrics = {}
            self.val_metrics = {}
            self.test_metrics = {}
        else:
            self.train_metrics = {'train_acc': Accuracy(task='multiclass', num_classes=2)}
            self.val_metrics = {'val_acc': Accuracy(task='multiclass', num_classes=2)}
            self.test_metrics = {'test_acc': Accuracy(task='multiclass', num_classes=2)}


        self.__set_attributes(self.train_metrics)
        self.__set_attributes(self.val_metrics)
        self.__set_attributes(self.test_metrics)

        self.__build_model()

    def __set_attributes(self, attributes_dict):
        for k,v in attributes_dict.items():
            setattr(self, k, v)

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.net = models.get_net(self.backbone)

        # 2. Loss:
        if self.loss_f == 'cross_entropy':
            self.loss_func = F.cross_entropy

        self.test_y = []
        self.test_y_hat = []

    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = torch.squeeze(self.forward(x))


        # 2. Compute loss:
        if self.loss_f == 'lr':
            train_loss = self.loss_func(y_logits, y, train=True)
        else:
            train_loss = self.loss_func(y_logits, y)

        # 3. Compute metrics and log:
        self.log("train_loss", train_loss, on_step=False, on_epoch=True,  prog_bar=True, logger=True, sync_dist=False)

        for metric_name in self.train_metrics.keys():
            self.log(metric_name, getattr(self, metric_name)(y_logits, y), on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=False)

        return train_loss

    def training_epoch_end(self, outputs):
        self.log("step", float(self.current_epoch), on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.train_metrics.keys():
            self.log(metric_name, getattr(self, metric_name).compute().cuda(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = torch.squeeze(self.forward(x))

        # 2. Compute loss:
        val_loss = self.loss_func(y_logits, y)

        # 3. Compute metrics and log:
        self.log('val_loss', val_loss, on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.val_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)

    def validation_epoch_end(self, outputs):
        self.log("step", float(self.current_epoch), on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.val_metrics.keys():
            self.log(metric_name, getattr(self, metric_name).compute().cuda(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()

    def test_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y, name = batch
        y_logits = torch.squeeze(self.forward(x))

        # 2. Compute loss:
        test_loss = self.loss_func(y_logits, y)

        # 3. Compute metrics and log:
        self.log('test_loss', test_loss, on_step=False, on_epoch=True,  prog_bar=False, logger=True, sync_dist=False)
        for metric_name in self.test_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)

    def test_epoch_end(self, outputs):
        test_summary = {'best_ckpt_path': self.trainer.resume_from_checkpoint}
        for metric_name in self.test_metrics.keys():
            test_summary[metric_name] = getattr(self, metric_name).compute()
            self.log(metric_name, test_summary[metric_name], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()

        return test_summary

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, name = batch
        logit = self(x)
        return y,name,logit



    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_name)

        optimizer_kwargs = {'momentum': 0.9} if self.optimizer_name == 'sgd' else {'eps': self.eps}

        optimizer = optimizer(self.parameters(),
                              lr=self.lr,
                              weight_decay=self.weight_decay,
                              **optimizer_kwargs)
        num_gpus = len(self.gpus.split(','))
        print('Number of gpus', num_gpus)

        steps_per_epochs = len(self.train_dataset)//num_gpus//self.batch_size
        self.warmup = self.warmup * steps_per_epochs # number of warmup epochs

        if self.lr_scheduler_name == 'cos':
            scheduler_kwargs = {'T_max': self.epochs*steps_per_epochs,
                                'eta_min':self.lr/50}


        elif self.lr_scheduler_name == 'onecycle':
            scheduler_kwargs = {'max_lr': self.lr, 'epochs': self.epochs, 'steps_per_epoch':steps_per_epochs,
                                'pct_start':3/self.epochs,'div_factor':25,
                                'final_div_factor':2}

        elif self.lr_scheduler_name == 'multistep':
             scheduler_kwargs = {'milestones':[50], 'gamma':0.5}

        elif self.lr_scheduler_name == 'lrdrop':
            scheduler_kwargs = dict(
                mode='min',
                factor=0.5,
                patience=4,
                verbose=False,
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0,
                min_lr=1e-7,
                eps=1e-08
            )

        scheduler = get_lr_scheduler(self.lr_scheduler_name)

        scheduler_params, interval = get_lr_scheduler_params(self.lr_scheduler_name, **scheduler_kwargs)

        scheduler = scheduler(optimizer, **scheduler_params)

        monitor = None
        if self.lr_scheduler_name == 'lrdrop':
            monitor = 'val_loss'
        return [optimizer], [{'scheduler': scheduler, 'monitor': monitor, 'interval': interval, 'name': 'lr'}]

    def lr_scheduler_step(self, scheduler, scheduler_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)




    def prepare_data(self):
        """Download images and prepare images datasets."""

        print('data downloaded')

    def setup(self, stage: str):


        folds = ['train', 'val']
        split = np.load('data/split.npy', allow_pickle=True).item()
        classes = list(split['labels'].keys())

        dataset = []
        for fold in folds:
            for cl in classes:
                for path in split[fold][cl]:
                    dataset.append({
                        'image_name': path,
                        'label': split['labels'][cl],
                        'fold':int(fold==folds[0]),
                        })

        random.shuffle(dataset)
        dataset = pd.DataFrame(dataset)

        self.train_dataset = Retriever(
            image_names=dataset[dataset['fold'] != 0].image_name.values,
            labels=dataset[dataset['fold'] != 0].label.values,
            transforms=get_train_transforms(self.imsize)
        )

        self.valid_dataset = Retriever(
            image_names=dataset[dataset['fold'] == 0].image_name.values,
            labels=dataset[dataset['fold'] == 0].label.values,
            transforms=get_valid_transforms(self.imsize)
        )


    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset

        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            collate_fn=None,
                            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='srnet',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--epochs',
                            default=50,
                            type=int,
                            metavar='N',
                            help='total number of epochs')
        parser.add_argument('--batch-size',
                            default=64,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--gpus',
                            type=str,
                            default='0',
                            help='which GPU to use')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-3,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--eps',
                            default=1e-7,
                            type=float,
                            help='eps for adaptive optimizers',
                            dest='eps')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--lr-scheduler-name',
                            default='lrdrop',
                            type=str,
                            metavar='LRS',
                            help='Name of LR scheduler')
        parser.add_argument('--optimizer-name',
                            default='adamax',
                            type=str,
                            metavar='OPTI',
                            help='Name of optimizer')
        parser.add_argument('--weight-decay',
                            default=1e-5,
                            type=float,
                            metavar='wd',
                            help='Optimizer weight decay')
        parser.add_argument('--loss_f',
                            default='cross_entropy',
                            type=str,
                            help='loss function')
        parser.add_argument('--warmup',
                            default=0,
                            type=int,
                            help='warmup epochs')
        parser.add_argument('--imsize',
                            default=512,
                            type=int,
                            help='image size')

        return parser
