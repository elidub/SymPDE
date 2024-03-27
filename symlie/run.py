import torch
import os
import pytorch_lightning as pl
import argparse
import logging
import wandb
import os, sys
import yaml
import ast

from model.setup import setup_model
from data.generate_2d import sine1d, sine2d, flower, mnist, noise
from data.generate_data import save_splits

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='SymLie')

    parser.add_argument("--config", type=str, default=None, help="Name of the config file")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--net", type=str, default='MLP', help="Name of the network")
    # parser.add_argument("--transform_type", type=str, default='space_translation', help="Type of the transformation")
    # parser.add_argument("--linearmodules", nargs='+', default=['MyLinearPw', 'nn.Linear'], help="Linearmodules")
    parser.add_argument("--bias", action="store_true", help="Bias")

    parser.add_argument("--criterion", type=str, default='mse', help="Criterion")
    parser.add_argument("--out_features", type=int, default=1, help="Out features")
    parser.add_argument("--n_classes", type=int, default=None, help="Number of classes")

    parser.add_argument("--data_dir", type=str, default="../data/sinev2", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Path to log directory")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=7, help="Number of workers")
    parser.add_argument("--version", type=str, default='version_0', help="Version of the training run")
    parser.add_argument("--name", type=str, default=None, help="Name of the training run")
    parser.add_argument("--model_summary", type=bool, default=False, help="Weights summary")

    # Data kwargs
    parser.add_argument("--grid_size", nargs='+', type=int, default = None) 
    parser.add_argument("--noise_std", type=float, default = None) 
    parser.add_argument("--y_low", type=int, default = None)
    parser.add_argument("--y_high", type=int, default = None)
    parser.add_argument("--A_low", type=float, default = None)
    parser.add_argument("--A_high", type=float, default = None)

    parser.add_argument("--grid_sizes", type=str, default = "[]") 
    parser.add_argument("--implicit_layer_dims", type=str, default = "[]") 
    parser.add_argument("--vanilla_layer_dims", nargs='+', type=int, default=None)


    parser.add_argument("--y_multi", type=int, default = 1)
    # Transformation kwargs
    parser.add_argument("--eps_mult", nargs='+', type=float, default=[1., 1., 1., 1.])
    parser.add_argument("--only_flip", type=bool, default=False)

    parser.add_argument("--use_P_from_noise", type = bool, default = False)

    parser.add_argument("--n_hidden_layers", type = int, default = 1)
    parser.add_argument("--svd_rank", type = int, default = None)

    parser.add_argument("--lossweight_y", type = float, default = 0.)
    parser.add_argument("--lossweight_o", type = float, default = 0.)
    parser.add_argument("--lossweight_dg", type = float, default = 0.)
    parser.add_argument("--lossweight_dx", type = float, default = 0.)
    parser.add_argument("--lossweight_do", type = float, default = 0.)
    parser.add_argument("--lossweight_do_tilde", type = float, default = 0.)
    parser.add_argument("--lossweight_do_tilde_mmd", type = float, default = 0.)
    # parser.add_argument("--lossweight_do_a", type = float, default = 0.)
    # parser.add_argument("--lossweight_do_b", type = float, default = 0.)
    # parser.add_argument("--lossweight_do_a_mmd", type = float, default = 0.)
    # parser.add_argument("--lossweight_do_a_mmd", type = float, default = 0.)

    parser.add_argument("--hidden_implicit_layers", nargs='+', type=int, default=None)

    # parser.add_argument("--persistent_workers", action="store_true", help="Persistent workers")
    parser.add_argument("--persistent_workers", default=True)

    parser.add_argument("--generate_data", action="store_true", help="Generate data")
    # parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run_id", type=str, default=None, help="Id of the training run")
    parser.add_argument("--tags", nargs='+', default=['dev'], help="Tags for wandb")
    parser.add_argument("--train", default=True)
    parser.add_argument("--predict", default=False)
    parser.add_argument("--test", default=True)
    parser.add_argument("--logger", default="wandb")
    parser.add_argument("--earlystop", action="store_true")
    
    parser.add_argument("--do_return", action="store_true", help="Return model, trainer, datamodule")
    parser.add_argument("--do_return_model", action="store_true", help="Return model, None, None")
    parser.add_argument("--do_return_datamodule", action="store_true", help="Return None, None, datamodule")

    # parser.add_argument("--n_splits", nargs='+', default=[10_000,5_000,5_000], help="Train, val, test split")
    parser.add_argument("--n_train", type=int, default=10_000, help="Train split")
    parser.add_argument("--n_val", type=int, default=1000, help="Val split")
    parser.add_argument("--n_test", type=int, default=1000, help="Test split")

    args = parser.parse_args([]) if notebook else parser.parse_args()


    #read yaml file
    config_dir = '../jobs/configs/'
    if args.config:
        with open(os.path.join(config_dir, f'{args.config}.yaml')) as file:
            yaml_config = yaml.safe_load(file)

        #update args with yaml file
        for key, value in yaml_config.items():
            setattr(args, key, value)

    return args

def printt(x):
    print(type(x), x)

def process_args(args):
    args.grid_size = tuple(args.grid_size) # Convert to tuple
    if isinstance(args.eps_mult, str): args.eps_mult = tuple([float(e_i) for e_i in args.eps_mult.split(' ')])
    if isinstance(args.eps_mult, list): args.eps_mult = tuple(args.eps_mult)

    if isinstance(args.vanilla_layer_dims, str): args.vanilla_layer_dims = [int(i) for i in args.vanilla_layer_dims.split(' ')]
    if isinstance(args.implicit_layer_dims, str): args.implicit_layer_dims = ast.literal_eval(args.implicit_layer_dims)
    if isinstance(args.grid_sizes, str): args.grid_sizes = ast.literal_eval(args.grid_sizes)

    data_kwargs_keys = ['grid_size', 'noise_std', 'y_low', 'y_high', 'A_low', 'A_high']
    args.data_kwargs = {k : getattr(args, k) for k in data_kwargs_keys}
    for data_kwargs_key in data_kwargs_keys:
        if args.data_kwargs[data_kwargs_key] is None:
            del args.data_kwargs[data_kwargs_key]

    if not isinstance(args.data_kwargs['grid_size'], tuple):
        print('Warning grid_size should already be tuple')
        args.data_kwargs['grid_size'] = tuple(args.data_kwargs['grid_size']) # Convert to tuple

    transform_kwargs_keys = ['eps_mult', 'only_flip']
    args.transform_kwargs = {k : getattr(args, k) for k in transform_kwargs_keys} 

    args.n_splits = [n_split for n_split in [args.n_train, args.n_val, args.n_test]]

    args.args_processed = True

    return args

def check_args_processed(args):
    if not hasattr(args, 'args_processed'):
        raise ValueError("Args not processed. Run `process_args(args)` first!")

def main(args):
    check_args_processed(args)    
    pl.seed_everything(args.seed, workers=True)

    if args.n_train >= 1000:
        args.max_epochs = 100

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, datamodule = setup_model(args)

    if args.do_return_datamodule:
        return None, None, datamodule

    if args.do_return_model:
        return model, None, None
    
    if args.run_id is None:
        if args.logger == "wandb":
            wandb.init(project="symlie", dir=args.log_dir, config=args, entity="eliasdubbeldam", tags=args.tags)
            logger = pl.loggers.WandbLogger(version=args.version, save_dir=args.log_dir, project = "symlie")
            enable_checkpointing = None
            print(f"Running {logger.experiment.name} with id {logger.version}")
        else:
            logger = False
            enable_checkpointing = False
            print(f"Running without logging")
    else:
        assert args.train is False, "Continue training not implemented"
        logger = False
        enable_checkpointing = False
        print(f"Loading {args.run_id}")

    # Disable lightning prints about GPU's 
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    os.environ["WANDB_SILENT"] = "true"

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        log_every_n_steps = 1,
        max_epochs=args.max_epochs,
        accelerator=args.device,
        deterministic=True,
        enable_model_summary=args.model_summary,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=False)] if args.earlystop else None,
    )  

    if args.do_return:
        args.train, args.test = False, False
        datamodule.setup()

    if args.train:
        trainer.fit(model, datamodule=datamodule)
    
    if args.test:
        trainer.test(model, datamodule=datamodule)

    wandb.finish()
    return model, trainer, datamodule

def generate_data(args):
    check_args_processed(args)   
     
    datasets = {
        "noise"  : {'create_sample_func' : noise},
        'sine1d' : {'create_sample_func' : sine1d},
        'sine1dtwo' : {'create_sample_func' : sine1d},
        'sine1dmtr' : {'create_sample_func' : sine1d},
        'sine2d' : {'create_sample_func' : sine2d},
        'flower' : {'create_sample_func' : flower},
        'MNIST'  : {'create_sample_func' : mnist},
    }
    
    dataset_key = args.data_dir.split('/')[-1]
    print(f"Generating data for {dataset_key}!")
    dataset = datasets[dataset_key]

    save_splits(
        create_sample_func = dataset['create_sample_func'],
        data_kwargs=args.data_kwargs,
        transform_kwargs=args.transform_kwargs,
        data_dir=args.data_dir,
        n_splits=args.n_splits
    )
    
if __name__ == '__main__':
    args = parse_options()
    args = process_args(args)
    print(args)

    # if args.generate_data:
    #     generate_data(args)
    # else:
    #     main(args)