import argparse
import os
import copy
import sys
import time
from datetime import datetime
import torch
import random
import numpy as np

from typing import Tuple
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys, os
sys.path.append(os.getcwd())
from common.utils import HDF5Dataset, DataCreator
from experiments.models_cnn import CNN, ResNet, BasicBlock1d
from experiments.models_fno import FNO1d
from experiments.train_helper import *
from equations.PDEs import PDE, KdV, KS, Heat
from common.augmentation import Subalgebra

import torch
torch.manual_seed(0)

import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../sympde'))
from data.dataset import PDEDataset, PDEDataModule

def check_directory() -> None:
    """
    Check if log directory exists within experiments.
    """
    if not os.path.exists(f'experiments/log'):
        os.mkdir(f'experiments/log')
    if not os.path.exists(f'models'):
        os.mkdir(f'models')

def train(args: argparse,
          pde: PDE,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          loader: DataLoader,
          data_creator: DataCreator,
          criterion: torch.nn.modules.loss,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done iteratively for the number of timesteps in one training sample.
    One average, we therefore start at every starting point in every trajectory during one episode.
    Args:
        args (argparse): command line input arguments
        pde (PDE): PDE at hand
        epoch (int): current training epoch
        model (torch.nn.Module): neural network model
        optimizer (torch.optim): chosen optimizer
        loader (DataLoader): train/valid/test dataloader
        data_creator (DataCreator): DataCreator to construct input data and labels
        criterion (torch.nn.modules.loss): loss criterion
        device: device (cpu/gpu)
    """
    print(f'Starting epoch {epoch}...')
    model.train()

    # Sample how many steps we use for unrolling our model (only need if pushforward trick is applied )
    # The default is to unroll zero steps in the first epoch and then increase the max amount of unrolling steps per additional epoch
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    # unrolling = [r for r in range(max_unrolling + 1)] # CHANGE but doesn't matter
    unrolling = [0]

    # Run every epoch (twice) as often as we have number of timesteps in one trajectory.
    # This is done iteratively for the number of timesteps in one training sample.
    # One average, we therefore start at every starting point in every trajectory during one episode (twice).
    for i in range(data_creator.t_res * 2): # CHANGE
    # for i in range(1):

        # losses = training_loop(pde, model, unrolling, args.batch_size, optimizer, loader, data_creator, criterion, device)
        data, labels = training_loop(pde, model, unrolling, args.batch_size, optimizer, loader, data_creator, criterion, device)
        return data, labels

        # if(i % args.print_interval == 0):
        #     print(f'Training Loss (progress: {i / (data_creator.t_res * 2):.2f}): {torch.mean(losses)}')
        print(f'Training Loss (progress: {i}): {torch.mean(losses)}')


def test(args: argparse,
         pde: PDE,
         model: torch.nn.Module,
         loader: DataLoader,
         data_creator: DataCreator,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test routine.
    Both step wise losses and enrolled forward losses are computed.
    Args:
        args (argparse): command line input arguments
        pde (PDE): PDE at hand
        model (torch.nn.Module): neural network model
        loader (DataLoader): train/valid/test dataloader
        data_creator (DataCreator): DataCreator to construct input data and labels
        criterion (torch.nn.modules.loss): loss criterion
        device: device (cpu/gpu)
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, torch. Tensor): mean and normalized mean errors
        of full trajectory unrolling
    """
    model.eval()

   # Check the losses for different timesteps (one forward prediction step)
    losses = test_timestep_losses(model=model,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  data_creator=data_creator,
                                  criterion=criterion,
                                  device=device)

    # Test the unrolled losses (full trajectory)
    losses, nlosses = test_unrolled_losses(model=model,
                                           batch_size=args.batch_size,
                                           nx=args.nx,
                                           loader=loader,
                                           data_creator=data_creator,
                                           criterion=criterion,
                                           device=device)


    mean, std = bootstrap(losses, 64, 1)
    nmean, nstd = bootstrap(nlosses, 64, 1)
    print(f'Unrolled forward losses: {mean:.4f} +- {std:.4f}')
    print(f'Unrolled forward losses (normalized): {nmean:.4f} +- {nstd:.4f}')
    return mean, std, nmean, nstd

def main(args: argparse):
    """
    Main method:
    Initialize equations and data augmentation,
    load corresponding datasets,
    initialize log file and save path of the model,
    initialize data creator and model,
    training loop
    Args:
        args (argparse): command line input arguments
    """
    device = args.device
    check_directory()

    # Initialize equations and data augmentation
    if args.experiment == 'KdV':
        pde = KdV()
        pde.time_shift = args.KdV_augmentation[0] # 1
        pde.max_x_shift = args.KdV_augmentation[1] # 1.
        pde.max_velocity = args.KdV_augmentation[2] # 0.4
        pde.max_scale = args.KdV_augmentation[3] # 0.1
    elif args.experiment == 'KS':
        pde = KS()
        pde.time_shift = args.KS_augmentation[0]
        pde.max_x_shift = args.KS_augmentation[1]
        pde.max_velocity = args.KS_augmentation[2]
    elif args.experiment == 'Burgers':
        # In the rest of the code, the naming "Heat" is used to make it easier.
        pde = Heat()
        pde.time_shift = args.Burgers_augmentation[0]
        pde.max_x_shift = args.Burgers_augmentation[1]
        pde.alpha = args.Burgers_augmentation[2]
        pde.subalgebra = Subalgebra(pde.nu, pde.alpha)
    else:
        raise Exception("Wrong experiment")

    # Load the corresponding datasets
    train_string = f'data/{pde}_train_{args.train_samples}.h5'
    valid_string = f'data/{pde}_valid.h5'
    test_string = f'data/{pde}_test.h5'
    if args.suffix:
        train_string = train_string.replace(f'.h5', f'_{args.suffix}.h5')
        valid_string = valid_string.replace(f'.h5', f'_{args.suffix}.h5')
        test_string = test_string.replace(f'.h5', f'_{args.suffix}.h5')
    try:
        # generator = True if args.KdV_augmentation[2] > 0 else False
        # print('Generator:', generator)
        # datamodule = PDEDataModule(
        #     pde_name = args.experiment, 
        #     data_dir = '../../data',
        #     batch_size = args.batch_size, 
        #     num_workers = args.num_workers,
        #     n_splits = [10, 10, 10],
        #     generators = generator,
        #     persistent_workers = True,
        # )

        # datamodule.setup()

        # train_loader = datamodule.train_dataloader()
        # valid_loader = datamodule.val_dataloader()
        # test_loader = datamodule.test_dataloader()

        # train_dataset = datamodule.train_dataloader().dataset
        # valid_dataset = datamodule.val_dataloader().dataset
        # test_dataset = datamodule.test_dataloader().dataset 

        mem = False

        train_dataset = HDF5Dataset(train_string,
                                    mode='train',
                                    nt=args.nt,
                                    nx=args.nx,
                                    shift=args.shift,
                                    pde=pde)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False, # TODO: shuffle=True
                                  num_workers=args.num_workers,
                                  persistent_workers=mem,
                                  pin_memory=mem)
        valid_dataset = HDF5Dataset(valid_string,
                                    mode='valid',
                                    nt=args.nt,
                                    nx=args.nx,
                                    shift=args.shift,
                                    pde=pde)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  persistent_workers=mem,
                                  pin_memory=mem)

        test_dataset = HDF5Dataset(test_string,
                                   mode='test',
                                   nt=args.nt,
                                   nx=args.nx,
                                   shift=args.shift,
                                   pde=pde)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 persistent_workers=mem, 
                                 pin_memory=mem)
    except:
        raise Exception("Datasets could not be loaded properly")
    
    # return train_dataset #CHECK1

    # print('dataset shapes', len(train_dataset), len(valid_dataset), len(test_dataset))
    print('dataloader shapes', len(train_loader), len(valid_loader), len(test_loader))

    # Log file and save path for model
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'

    aug_string = ''
    if str(pde) == 'KdV':
        aug_string = f'{int(pde.time_shift == 1)}{int(pde.max_x_shift > 0.)}' \
                     f'{int(pde.max_velocity > 0.)}{int(pde.max_scale > 0.)}'
    elif str(pde) == 'KS':
        aug_string = f'{int(pde.time_shift == 1)}{int(pde.max_x_shift > 0.)}' \
                     f'{int(pde.max_velocity > 0.)}'
    elif str(pde) == 'Heat':
        aug_string = f'{int(pde.time_shift == 1)}{int(pde.max_x_shift > 0.)}' \
                     f'{int(pde.alpha > 0.)}'
    else:
        raise Exception("Datasets could not be loaded properly")

    if(args.log):
        logfile = f'experiments/log/{args.model}_{pde}_samples{args.train_samples}_augmentation{aug_string}' \
                  f'_shift{args.shift}_future{args.time_future}_time{timestring}.csv'
        if args.unrolling > 0:
            logfile = logfile.replace(f'_time', f'_unrolling{args.unrolling}_time')
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')
    save_path = f'models/{args.model}_{pde}_samples{args.train_samples}_augmentation{aug_string}_shift{args.shift}' \
                f'_future{args.time_future}_time{timestring}.pt'
    if args.unrolling > 0:
        save_path = save_path.replace(f'_time', f'_unrolling{args.unrolling}_time')
    print(f'Training on dataset {train_string}')
    print(save_path)

    # Initialize DataCreator and model
    data_creator = DataCreator(time_history=args.time_history,
                               time_future=args.time_future,
                               t_resolution=args.nt,
                               x_resolution=args.nx
                               ).to(device)

    if args.model == 'FNO1d':
        model = FNO1d(pde=pde,
                      time_history=args.time_history,
                      time_future=args.time_future).to(device)
    elif args.model == 'CNN':
        model = CNN(pde=pde,
                    time_history=args.time_history,
                    time_future=args.time_future).to(device)
    elif args.model == 'ResNet':
        # 1D Res18 architecture
        model = ResNet(pde,
                       BasicBlock1d,
                       [2, 2, 2, 2],
                       time_history=args.time_history,
                       time_future=args.time_future).to(device)
    else:
        raise Exception("Wrong model specified")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    # return dict(train_dataset=train_dataset, model=model)

    # Optimizer
    # if args.optimizer == 'adamw':
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr) # CHANGE
    # elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    # else:
    #    raise Exception("Wrong optimizer specified")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.unrolling, 5, 10, 15], gamma=args.lr_decay)

    # Training loop
    min_val_loss = 10e30
    test_loss, ntest_loss = 10e30, 10e30
    test_loss_std, ntest_loss_std = 0., 0.
    criterion = torch.nn.MSELoss(reduction="none")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        data, labels = train(args, pde, epoch, model, optimizer, train_loader, data_creator, criterion, device=device)
        return dict(train_dataset=train_dataset, train_loader=train_loader,model=model, data=data, labels=labels)
        
        
        print("Evaluation on validation dataset:")
        val_loss, _, _, _ = test(args, pde, model, valid_loader, data_creator, criterion, device=device)
        if(val_loss < min_val_loss):
            print("Evaluation on test dataset:")
            test_loss, test_loss_std, ntest_loss, ntest_loss_std = test(args,
                                                                         pde,
                                                                         model,
                                                                         test_loader,
                                                                         data_creator,
                                                                         criterion,
                                                                         device=device)
            # Save model
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at {save_path}\n")
            min_val_loss = val_loss

        # if args.scheduler:
        # scheduler.step() # CHANGE

    print(f'Test loss mean {test_loss:.4f}, test loss std: {test_loss_std:.4f}')
    print(f'Normalized test loss mean {ntest_loss:.4f}, normalized test loss std: {ntest_loss_std:.4f}')

def parse_options(notebook = False):
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer, has to be adamw or adam')
    parser.add_argument('--scheduler', action = 'store_true', default = False, help = 'Use scheduler')

    # PDE
    parser.add_argument('--device', type=str, default='cuda',
                        help='Used device')
    parser.add_argument('--experiment', type=str, default='KdV',
                        help='Experiment for PDE solver should be trained: [KdV, KS, Burgers]')
    parser.add_argument('--KdV_augmentation', type=lambda s: [float(item) for item in s.split(',')],
                        default=[0, 0.0, 0.0, 0.0], help="[time_shift, max_x_shift, max_velocity, max_eps]")
    parser.add_argument('--KS_augmentation', type=lambda s: [float(item) for item in s.split(',')],
                        default=[0, 0.0, 0.0], help="[time_shift, max_x_shift, max_velocity]")
    parser.add_argument('--Burgers_augmentation', type=lambda s: [float(item) for item in s.split(',')],
                        default=[0, 0.0, 0.0], help="[time_shift, max_x_shift, max_alpha]")
    parser.add_argument('--train_samples', type=int, default=512,
                        help='Number of training samples')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for additional datasets')
    parser.add_argument('--shift', type=str,
                        default='fourier', help="Fourier or linear shift")
    # Model
    parser.add_argument('--model', type=str, default='FNO1d',
                        help='Model used as PDE solver: [FNO1d, ResNet, CNN]')
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float,
                        default=0.4, help='multistep lr decay')
    # Misc
    parser.add_argument('--nt', type=int, default=140,
                        help="Temporal resolution")
    parser.add_argument('--nx', type=int, default=256,
                        help="Spatial resolution")
    parser.add_argument('--time_history', type=int,
                        default=20, help="Time steps to be considered as input to the solver")
    parser.add_argument('--time_future', type=int,
                        default=20, help="Time steps to be considered as output of the solver")
    parser.add_argument('--unrolling', type=int,
                        default=0, help="Unrolling which proceeds with each epoch")
    parser.add_argument('--print_interval', type=int, default=20,
                        help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
                        help='pip the output to log file')
    parser.add_argument('--num_workers', type=int, default=4,)

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_options()
    main(args)
