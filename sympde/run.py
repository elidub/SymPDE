import torch
import os
import pytorch_lightning as pl
import argparse
import logging


from data.dataset import PDEDataset, PDEDataModule
from model.setup import setup_model
from data.lpda_data_aug import SpaceTranslate, Scale, Galileo

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymPDE')

    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--pde_name", type=str, default="pde1", help="Name of the PDE")
    parser.add_argument("--net", type=str, default='FNO1d', help="Name of the network")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--version", type=str, default=None, help="Version of the training run")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--local", action="store_true", help="Run on local machine")
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")

    parser.add_argument("--n_splits", nargs='+', default=[-1,-1,-1], help="Train, val, test split")
    parser.add_argument("--generators", action="store_true", help="Use generators")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    if args.version == None:
        generator_indicator = '1' if args.generators else '0'
        args.version = f'precision_aug{generator_indicator}_{args.pde_name}_seed{args.seed}'
        print("\n\n###Version: ", args.version, "###\n\n")

    datamodule = PDEDataModule(
        pde_name = args.pde_name, 
        data_dir = args.data_dir, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers,
        n_splits = [int(n_split) for n_split in args.n_splits],
        generators = args.generators,
        persistent_workers = args.persistent_workers,
    )

    model = setup_model(args)

    logging.getLogger('lightning').setLevel(0) # Disable lightning prints about GPU's
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            args.log_dir, name=args.net, version=args.version
        ),
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