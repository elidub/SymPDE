import os
import torch.nn as nn
import torch

from model.learner import Learner
from model.loss import LpLoss

from model.networks.fno import FNO1d
from model.networks.cnn import CNN, ResNet, BasicBlock1d
from model.networks.mlp import MLP

def setup_model(args):
    net = args.net

    time_history = 10
    time_future  = 5

    if net == "FNO1d":
        net = FNO1d(time_history=time_history, time_future=time_future)
    elif net == "CNN":
        net = CNN(time_history=time_history, time_future=time_future)
    elif net == "ResNet":
        net = ResNet(BasicBlock1d, [2, 2, 2, 2], time_history=time_history, time_future=time_future)
    elif net == "MLP":
        hidden_channels = [100, 100, 100] if args.mlp_hidden_channels is None else args.mlp_hidden_channels
        net = MLP(time_history=time_history, time_future=time_future, hidden_channels=hidden_channels)
    else:
        raise NotImplementedError
    
    criterion = LpLoss()

    if args.train:
        model = Learner(net, criterion)
        return model
    
    # Load model
    ckpt_path = os.path.join(args.log_dir, args.name, args.version, "checkpoints")
    assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found"
    ckpt = os.listdir(ckpt_path)[0]
    model = Learner.load_from_checkpoint(
        os.path.join(ckpt_path, ckpt), net=net, criterion=criterion,
        map_location=torch.device('cpu')
    )

    return model
