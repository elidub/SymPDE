import os
import torch.nn as nn
import torch

from model.networks.fno import FNO1d
from model.learner import Learner
from model.loss import LpLoss

def setup_model(args):
    net = args.net

    if net == "FNO1d":

        time_history = 10
        time_future  = 5

        net = FNO1d(
            time_history = time_history,
            time_future  = time_future,
        )
        criterion = LpLoss()
    else:
        raise NotImplementedError
    

    if args.train:
        model = Learner(net, criterion)
        return model
    
    ckpt_path = os.path.join(args.log_dir, args.net, args.version, "checkpoints")
    assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found"
    ckpt = os.listdir(ckpt_path)[0]
    model = Learner.load_from_checkpoint(
        os.path.join(ckpt_path, ckpt), net=net, criterion=criterion,
        map_location=torch.device('cpu')
    )

    return model
