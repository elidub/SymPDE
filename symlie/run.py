import torch
import os
import pytorch_lightning as pl
from icecream import install
import argparse
import logging
import math
import json

install()

from model.setup import setup_model
from data.generate_flower import create_flower
from data.generate_data import save_splits

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymLie')

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--net", type=str, default='MLP', help="Name of the network")
    # parser.add_argument("--transform_type", type=str, default='space_translation', help="Type of the transformation")
    # parser.add_argument("--linearmodules", nargs='+', default=['MyLinearPw', 'nn.Linear'], help="Linearmodules")
    parser.add_argument("--bias", action="store_true", help="Bias")

    parser.add_argument("--data_dir", type=str, default="../data/sinev2", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, default="../logs_symlie", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=7, help="Number of workers")
    parser.add_argument("--version", type=str, default=None, help="Version of the training run")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run")
    parser.add_argument("--model_summary", type=bool, default=False, help="Weights summary")

    # Data kwargs
    parser.add_argument("--space_length", type=int, default = 7) # 7    7
    parser.add_argument("--noise_std", type=float, default= 0.01)  # 0.5  0.01
    parser.add_argument("--y_low", type=int, default = 1)
    parser.add_argument("--y_high", type=int, default = 3)

    # Transformation kwargs
    parser.add_argument("--eps_mult", type=list, default=[0., 0., 1., 1.])
    parser.add_argument("--only_flip", type=bool, default=True)

    # parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--persistent_workers", default=True)

    parser.add_argument("--generate_data", action="store_true", help="Generate data")
    # parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--train", default=True)
    parser.add_argument("--predict", default=True)
    parser.add_argument("--test", default=True)
    
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")
    parser.add_argument("--do_return_model", action="store_true", help="Return model, None, None")

    parser.add_argument("--n_splits", nargs='+', default=[100_000,5_000,5_000], help="Train, val, test split")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def process_args(args):
    data_kwargs_keys = ['space_length', 'noise_std', 'y_low', 'y_high']
    args.data_kwargs = {k : getattr(args, k) for k in data_kwargs_keys}

    transform_kwargs_keys = ['eps_mult', 'only_flip']
    args.transform_kwargs = {k : getattr(args, k) for k in transform_kwargs_keys} 

    args.n_splits = [int(n_split) for n_split in args.n_splits]

    # Check if name is in args
    if hasattr(args, 'name'): 
        if args.name is None:
            data_dir = args.data_dir.split('/')[-1]
            bias = str(args.bias).lower()
            args.name = f'symlieflat_data{data_dir}_net{args.net}_lr{math.log10(args.lr):.2f}_seed{args.seed}_ntrain{args.n_splits[0]}_noise{args.noise_std}'

    args.args_processed = True

    return args

def check_args_processed(args):
    if not hasattr(args, 'args_processed'):
        if not hasattr(args, 'args_processed'):
            raise ValueError("Args not processed. Run process_args(args) first")

def main(args):
    check_args_processed(args)    
    pl.seed_everything(args.seed, workers=True)

    logger = pl.loggers.TensorBoardLogger(args.log_dir, name=args.name, version=args.version)
    args.version = logger.version

    print("\n\n###\tVersion: ", args.version, "\t###\n###\tName: ", args.name, "\t###\n\n")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, datamodule = setup_model(args)

    if args.do_return_model:
        return model, None, None, None

    # Disable lightning prints about GPU's 
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps = 1,
        max_epochs=args.max_epochs,
        accelerator=args.device,
        deterministic=True,
        enable_model_summary=args.model_summary,
    )  

    if args.do_return:
        datamodule.setup()
        return model, trainer, datamodule, None

    if args.train:
        trainer.fit(model, datamodule=datamodule)
    
    if args.test:
        trainer.test(model, datamodule=datamodule)
        return model, trainer, datamodule, None

    if args.predict:
        print("Predicting...")
        preds = trainer.predict(model, datamodule=datamodule)

        y_trues, y_preds = zip(*preds)        
        y_trues, y_preds = torch.cat(y_trues), torch.cat(y_preds)
        torch.save(y_trues, os.path.join(logger.log_dir, 'y_trues.pt'))
        torch.save(y_preds, os.path.join(logger.log_dir, 'y_preds.pt'))
        return model, trainer, datamodule, preds

    raise NotImplementedError("No action specified")

    return model, trainer, datamodule, None

def generate_data(args):
    check_args_processed(args)    
    create_sample_func_dict = {
        'flower': create_flower
    }

    save_splits(
        # create_sample_func=create_sample_func_dict[args.data_name],
        create_sample_func = create_flower,
        data_kwargs=args.data_kwargs,
        transform_kwargs=args.transform_kwargs,
        data_dir=args.data_dir,
        n_splits=args.n_splits
    )
if __name__ == '__main__':
    args = parse_options()
    args = process_args(args)
    print(args)

    if args.generate_data:
        generate_data(args)
    else:
        main(args)