import torch
import os
import pytorch_lightning as pl
import argparse
import logging
import math

from data.dataset import MnistDataModule, FlatDataModule
from model.setup import setup_model

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymLie')

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--net", type=str, default='MLP', help="Name of the network")
    parser.add_argument("--transform_type", type=str, default='space_translation', help="Type of the transformation")
    parser.add_argument("--linearmodules", nargs='+', default=['MyLinearPw', 'nn.Linear'], help="Linearmodules")
    parser.add_argument("--bias", action="store_true", help="Bias")

    parser.add_argument("--data_kwargs", type=dict, default = {
                            'space_length': 7,
                            'noise_std': 0.5,
                            'y_low': 1,
                            'y_high': 3,
                        }, help="Data kwargs")

    parser.add_argument("--data_dir", type=str, default="../data/flat", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=7, help="Number of workers")
    parser.add_argument("--version", type=str, default=None, help="Version of the training run")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run")

    # parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--persistent_workers", default=True)
    # parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--train", default=True)
    parser.add_argument("--local", action="store_true", help="Run on local machine")
    
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")
    parser.add_argument("--do_return_model", action="store_true", help="Return model, None, None")

    parser.add_argument("--n_splits", nargs='+', default=[400,1000,1000], help="Train, val, test split")

    # parser.add_argument("--dim", type=int, default=4, help="Dimension of the image")


    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    args.n_splits = [int(n_split) for n_split in args.n_splits]

    if args.name is None:
        data_dir = args.data_dir.split('/')[-1]
        bias = str(args.bias).lower()
        args.name = f'symlie_data{data_dir}_net{args.net}_transform{args.transform_type}_lr{math.log10(args.lr):.2f}_seed{args.seed}'
    print("\n\n###\tVersion: ", args.version, "\t###\n###\tName: ", args.name, "\t###\n\n")

    # datamodule = MnistDataModule(
    # datamodule = FlatDataModule(
    #     dim = args.dim,
    #     augment = args.transform_type,
    #     # augment_kwargs = {},
    #     data_dir = args.data_dir, 
    #     batch_size = args.batch_size,
    #     num_workers = args.num_workers,
    #     n_splits = args.n_splits,
    #     persistent_workers = args.persistent_workers,
    # )


    model, datamodule = setup_model(args)

    if args.do_return_model:
        return model, None, None

    logging.getLogger('lightning').setLevel(0) # Disable lightning prints about GPU's
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            args.log_dir, name=args.name, version=args.version
        ),
        log_every_n_steps = 1,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        deterministic=True,
    )  

    if args.do_return:
        datamodule.setup()
        return model, trainer, datamodule

    trainer.test(model, datamodule=datamodule)

    if args.train:
        trainer.fit(model, datamodule=datamodule)
    
    trainer.test(model, datamodule=datamodule)

    if hasattr(args, 'predict'):
        if args.predict:
            preds = trainer.predict(model, datamodule=datamodule)
            return model, trainer, datamodule, preds

    return model, trainer, datamodule

if __name__ == '__main__':
    args = parse_options()
    print(args)
    model, trainer, datamodule = main(args)

    next(iter(datamodule.test_dataloader()))
