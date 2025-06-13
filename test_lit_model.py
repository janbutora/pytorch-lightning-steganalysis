"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from argparse import ArgumentParser
from LitModel import *
from retriever import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from tools.log_utils import setup_callbacks_loggers
import numpy as np

seed_everything(1994)

def main(args):
    """ Main training routine specific for this project. """

    model = LitModel(**vars(args))
    model = LitModel.load_from_checkpoint(checkpoint_path)


    ckpt_callback, loggers, lr_logger = setup_callbacks_loggers(args)

    trainer = Trainer(logger=loggers,
                     callbacks=[ckpt_callback, lr_logger],
                     accelerator='gpu',
                     strategy='ddp' if len(args.gpus or '') > 1 else None,
                     devices=[int(g) for g in args.gpus.split(',')],
                     precision=16,
                     amp_backend='apex',
                     amp_level='O1',
                     log_every_n_steps=100,
                     benchmark=True,
                     sync_batchnorm=len(args.gpus or '') > 1,
                     resume_from_checkpoint=args.ckpt_path)


    folds = ['test']
    split = np.load('data/split.npy', allow_pickle=True).item()
    classes = list(split['labels'].keys())

    dataset = []
    for fold in folds:
        for cl in classes:
            for path in split[fold][cl]:
                dataset.append({
                    'image_name': path,
                    'label': split['labels'][cl],
                    'fold':2,
                    })

    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)

    test_dataset = Retriever(
            image_names=dataset[dataset['fold'] == 2].image_name.values,
            labels=dataset[dataset['fold'] == 2].label.values,
            transforms=get_valid_transforms(model.imsize),
            return_name=True,
        )

    test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    trainer.test(model, test_dataloader, args.ckpt_path)


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

    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    run_cli()
