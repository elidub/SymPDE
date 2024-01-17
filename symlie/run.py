import torch
import os
import pytorch_lightning as pl
import argparse
import logging
import math
import json

from model.setup import setup_model

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymLie')

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--net", type=str, default='MLP', help="Name of the network")
    # parser.add_argument("--transform_type", type=str, default='space_translation', help="Type of the transformation")
    # parser.add_argument("--linearmodules", nargs='+', default=['MyLinearPw', 'nn.Linear'], help="Linearmodules")
    parser.add_argument("--bias", action="store_true", help="Bias")

    parser.add_argument("--data_kwargs", default = {
                            'space_length': 7,
                            'noise_std': 0.5,
                            'y_low': 1,
                            'y_high': 3,
                        }, help="Data kwargs", type=json.loads)

    parser.add_argument("--data_dir", type=str, default="../data/flat", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=7, help="Number of workers")
    parser.add_argument("--version", type=str, default=None, help="Version of the training run")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run")
    parser.add_argument("--model_summary", type=bool, default=False, help="Weights summary")


    # parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--persistent_workers", default=True)
    # parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--train", default=True)
    parser.add_argument("--predict", default=True)
    parser.add_argument("--test", default=False)
    
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")
    parser.add_argument("--do_return_model", action="store_true", help="Return model, None, None")

    parser.add_argument("--n_splits", nargs='+', default=[400,1000,1000], help="Train, val, test split")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    args.data_kwargs = {k : float(v) for k, v in args.data_kwargs.items()}
    # data_kwargs = json.loads(args.data_kwargs)
    # print(data_kwargs)
    print(args.data_kwargs)
    return None, None, None, None
    pl.seed_everything(args.seed, workers=True)

    args.n_splits = [int(n_split) for n_split in args.n_splits]

    if args.name is None:
        data_dir = args.data_dir.split('/')[-1]
        bias = str(args.bias).lower()
        args.name = f'symlieflat_data{data_dir}_net{args.net}_lr{math.log10(args.lr):.2f}_seed{args.seed}'
    print("\n\n###\tVersion: ", args.version, "\t###\n###\tName: ", args.name, "\t###\n\n")

    model, datamodule = setup_model(args)

    if args.do_return_model:
        return model, None, None, None

    # Disable lightning prints about GPU's 
    # logging.getLogger('lightning').setLevel(0) 
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            args.log_dir, name=args.name, version=args.version
        ),
        log_every_n_steps = 1,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
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

    if args.predict:
        print("Predicting...")
        preds = trainer.predict(model, datamodule=datamodule)
        return model, trainer, datamodule, preds

    raise NotImplementedError("No action specified")

    return model, trainer, datamodule, None

if __name__ == '__main__':
    args = parse_options()
    print(args)
    main(args)