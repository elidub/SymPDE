import torch
import os
import pytorch_lightning as pl
import argparse
import logging

from lconv.mnist import MnistDataModule
from lconv.model import LconvNet
from lconv.learner import LconvLearner

from data.dataset import PDEDataset, PDEDataModule
from model.setup import setup_model

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymPDE')

    parser.add_argument("--data_dir", type=str, default="../data/mnist", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--net", type=str, default='FNO1d', help="Name of the network")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--version", type=str, default=None, help="Version of the training run")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")

    parser.add_argument("--n_splits", nargs='+', default=[-1,-1,-1], help="Train, val, test split")
    parser.add_argument("--rot_angle", default = 45, type = float, help="Rotation angle")
    parser.add_argument("--dim", default = 5, type = int, help="Image dimension")
    parser.add_argument("--digit", default = None, type = int, help="Digit to train on")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    args.n_splits = [int(n_split) for n_split in args.n_splits]

    if args.name is None:
        data_dir = args.data_dir.split('/')[-1]
        digit = args.digit if args.digit is not None else 'all'
        args.name = f'lconv_mnist_dim{args.dim}_r{args.rot_angle}_digit{digit}'
    print("\n\n###\tVersion: ", args.version, "\t###\n###\tName: ", args.name, "\t###\n\n")

    datamodule = MnistDataModule(
        r = args.rot_angle,
        dim = args.dim,
        digit = args.digit,
        data_dir = args.data_dir, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        n_splits = args.n_splits,
        persistent_workers = args.persistent_workers,
    )

    model = LconvLearner(
        net = LconvNet(shape = torch.tensor((1, 1, args.dim, args.dim))),
        criterion = torch.nn.MSELoss(),
        r = args.rot_angle,
        dim = args.dim,
    )                 

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

    if args.train:
        trainer.fit(model, datamodule=datamodule)
    
    trainer.test(model, datamodule=datamodule)

    return model, trainer, datamodule

if __name__ == '__main__':
    args = parse_options()
    print(args)
    main(args)