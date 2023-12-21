import os
import torch.nn as nn
import torch

from model.learner import Learner

from model.networks.mlp import MLP, MLPTorch
from model.networks.linear import MyLinear

def setup_model(args):
    net = args.net
    space_length = args.dim

    linearmodules = []
    for linearmodule in args.linearmodules:
        if linearmodule == "MyLinear":
            linearmodules.append(MyLinear)
        elif linearmodule == "nn.Linear":
            linearmodules.append(nn.Linear)
        else:
            raise NotImplementedError(f"Linearmodule {linearmodule} not implemented") 

    if net == "MLP":
        net = MLP(transform_type=args.transform_type,space_length=space_length, linearmodules=linearmodules, bias=args.bias)
    elif net == "MLPTorch":
        net = MLPTorch(space_length=space_length)
    else:
        raise NotImplementedError(f"Network {net} not implemented")
    
    criterion = nn.CrossEntropyLoss()

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
