import os
import torch.nn as nn
import torch

from model.networks.fno import FNO1d
from model.learner import Learner
from model.loss import LpLoss

def setup_model(args):
    net = args.net
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW
    else:
        raise NotImplementedError
    
    lr = args.lr
    
    if args.scheduler == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR
    else:
        scheduler = None

    if net == "FNO1d":

        time_history = args.time_history
        time_future  = args.time_future

        net = FNO1d(
            time_history = time_history,
            time_future  = time_future,
        )
        criterion = LpLoss()
    else:
        raise NotImplementedError
    

    if args.train:
        model = Learner(
            net=net, criterion=criterion, optimizer=optimizer, lr=lr, scheduler=scheduler,
        )
        return model
    
    ckpt_path = os.path.join(args.log_dir, args.net, args.version, "checkpoints")
    assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found"
    ckpt = os.listdir(ckpt_path)[0]
    model = Learner.load_from_checkpoint(
        os.path.join(ckpt_path, ckpt), 
            net=net, criterion=criterion, optimizer=optimizer, lr=lr, scheduler=scheduler,
        map_location=torch.device('cpu')
    )

    return model
