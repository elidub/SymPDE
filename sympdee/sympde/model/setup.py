import os
import torch.nn as nn
import torch

from model.learner import Learner
from model.loss import LpLoss

from model.networks.fno import FNO1d
from model.networks.cnn import CNN, ResNet, BasicBlock1d, ResNet_conv
from model.networks.mlp import MLP, CustomMLP
from model.networks.mlp_flat import MLPFlat

def setup_model(args):
    net = args.net
    assert False

    space_length = 256

    if net == "FNO1d":
        net = FNO1d(time_history=args.time_history, time_future=args.time_future)
    elif net == "CNN":
        net = CNN(time_history=args.time_history, time_future=args.time_future, embed_spacetime=args.embed_spacetime)
    elif net == "ResNet":
        net = ResNet(BasicBlock1d, [2, 2, 2, 2], time_history=args.time_history, time_future=args.time_future, embed_spacetime=args.embed_spacetime)
    elif net == "ResNet_conv":
        net = ResNet_conv(BasicBlock1d, [2, 2, 2, 2], time_history=args.time_history, time_future=args.time_future, embed_spacetime=args.embed_spacetime, equiv = args.equiv)
    elif net == "MLP":
        hidden_channels = [100, 100, 100] if args.mlp_hidden_channels is None else args.mlp_hidden_channels
        net = MLP(time_history=args.time_history, time_future=args.time_future, hidden_channels=hidden_channels)
    elif net == "CustomMLP":
        hidden_channels = [100, 100, 100] if args.mlp_hidden_channels is None else args.mlp_hidden_channels
        net = CustomMLP(time_history=args.time_history, time_future=args.time_future, hidden_channels=hidden_channels, embed_spacetime=args.embed_spacetime, equiv = args.equiv)
    elif net == "MLPFlat":
        hidden_channels = [100, 100, 100] if args.mlp_hidden_channels is None else args.mlp_hidden_channels
        net = MLPFlat(time_history=args.time_history, time_future=args.time_future, space_length = space_length, hidden_channels=hidden_channels, embed_spacetime=args.embed_spacetime)
    else:
        raise NotImplementedError(f"Network {net} not implemented")
    
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
