"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from argparse import ArgumentParser
from LitModel import *
import torch
from pytorch_lightning import Trainer, seed_everything
from tools.log_utils import setup_callbacks_loggers

seed_everything(666)

def main(args):
    """ Main training routine specific for this project. """
    model = LitModel(**vars(args))
    ckpt_callback, loggers, lr_logger = setup_callbacks_loggers(args)

    trainer = Trainer(logger=loggers,
                     callbacks=[ckpt_callback, lr_logger],
                     accelerator='gpu',
                     strategy='ddp' if len(args.gpus or '') > 1 else None,
                     devices=[int(g) for g in args.gpus.split(',')],
                     min_epochs=args.epochs,
                     max_epochs=args.epochs,
                     precision=16,
                     log_every_n_steps=100,
                     benchmark=True,
                     sync_batchnorm=len(args.gpus or '') > 1,
                     accumulate_grad_batches=args.accumulate,
                     enable_progress_bar=True)

    if args.seed is not None:
        print('Seeding from', args.seed)
        checkpoint = torch.load(args.seed, map_location='cuda:'+str(torch.cuda.current_device()))
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.net.load_state_dict(checkpoint)



    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model, ckpt_path=args.ckpt_path)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)

    parser.add_argument('--version',
                         default=None,
                         type=str,
                         metavar='V',
                         help='version or id of the net')
    parser.add_argument('--ckpt-path',
                         default=None,
                         type=str,
                         metavar='RFC',
                         help='path to checkpoint')
    parser.add_argument('--seed',
                            default=None,
                            type=str,
                            help='path to seeding checkpoint')
    parser.add_argument('--accumulate',
                            default=1,
                            type=int,
                            help='accumulate grad batches')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
