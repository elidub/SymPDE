import torch
import os
import pytorch_lightning as pl
import argparse
import logging


from data.dataset import PDEDataset, PDEDataModule
from model.setup import setup_model

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymPDE')

    parser.add_argument("--data_dir", type=str, default="../data/v1", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--pde_name", type=str, default="KdV", help="Name of the PDE")
    parser.add_argument("--net", type=str, default='FNO1d', help="Name of the network")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--version", type=str, default=None, help="Version of the training run")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--local", action="store_true", help="Run on local machine")
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")

    parser.add_argument("--n_splits", nargs='+', default=[-1,-1,-1], help="Train, val, test split")
    parser.add_argument("--epsilons", nargs='+', default=[], help="Epsilons for the generators")

    parser.add_argument("--mlp_hidden_channels", nargs='+', default=None, help="Hidden channels for MLP")

    # Model setup args
    parser.add_argument("--time_history", type=int, default = 10, help = "Time steps passed to network")
    parser.add_argument("--time_future",  type=int, default = 5,  help = "Time steps to predict by network")
    parser.add_argument("--embed_spacetime", action ="store_true", help = "Concatenate dx and dt to u in network")
    parser.add_argument("--equiv", type = str, default = "none", help = "Type of equivariance to use (none, mag)")


    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    args.epsilons = [float(eps) for eps in args.epsilons]
    args.n_splits = [int(n_split) for n_split in args.n_splits]
    args.mlp_hidden_channels = [int(hidden_channel) for hidden_channel in args.mlp_hidden_channels] if args.mlp_hidden_channels is not None else None

    if args.name is None:
        epsilons = '-'.join([str(eps) for eps in args.epsilons]) if len(args.epsilons) > 0 else '0'
        data_dir = args.data_dir.split('/')[-1]
        args.name = f'data{data_dir}_net{args.net}_equiv{args.equiv}_{args.pde_name}_aug{epsilons}_seed{args.seed}'
        if args.mlp_hidden_channels is not None:
            mlp_hidden_channels = '-'.join([str(hidden_channel) for hidden_channel in args.mlp_hidden_channels])
            args.name += f'_mlp{mlp_hidden_channels}'
    print("\n\n###\tVersion: ", args.version, "\t###\n###\tName: ", args.name, "\t###\n\n")

    datamodule = PDEDataModule(
        pde_name = args.pde_name, 
        data_dir = args.data_dir, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers,
        n_splits = args.n_splits,
        epsilons = args.epsilons,
        persistent_workers = args.persistent_workers,
    )

    model = setup_model(args)

    logging.getLogger('lightning').setLevel(0) # Disable lightning prints about GPU's
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            args.log_dir, name=args.name, version=args.version
        ),
        log_every_n_steps = 1,
        enable_checkpointing = False, # Don't save checkpoints, for debugging only
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