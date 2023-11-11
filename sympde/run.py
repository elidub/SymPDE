import torch
import pytorch_lightning as pl
import argparse

from data.dataset import PDEDataset, PDEDataModule
from model.setup import setup_model


def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymPDE')

    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--pde_name", type=str, default="pde1", help="Name of the PDE")
    parser.add_argument("--net", type=str, default='FNO1d', help="Name of the network")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--version", type=str, default="v0", help="Version of the training run")

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    datamodule = PDEDataModule(
        pde_name = args.pde_name, 
        data_dir = args.data_dir, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers
    )

    model = setup_model(args)

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            args.log_dir, name=args.net, version=args.version
        ),
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks = [
            pl.callbacks.TQDMProgressBar(refresh_rate=1000),
        ],
        deterministic=True
    )  

    trainer.fit(model, datamodule=datamodule)





if __name__ == '__main__':
    args = parse_options()
    print(args)
    main(args)